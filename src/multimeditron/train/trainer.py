import os
import torch
from torch.utils.data import DataLoader
from typing import List
from enum import IntEnum
from transformers import Trainer 
from accelerate import Accelerator
from typing import Optional, Any, Dict, Union
import warnings


if os.environ.get('ENABLE_BENCHY', None) == '1':
    from benchy.torch import BenchmarkGenericIteratorWrapper


class TrainingMode(IntEnum):
    ALIGNMENT = 0
    END2END = 1
    LM_ONLY = 2
    FULL = 3


TRAINING_MAPPING = {i.name: i for i in TrainingMode}


class MultimodalTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        # on_the_fly_embedding: bool = True,
        callbacks=None,
        optimizers=(None, None),
        training_mode: TrainingMode = TrainingMode.ALIGNMENT,
        pytorch_profiler_config=None,
        **kwargs
    ):
        """
        Initializes the trainer.

        Args:
            model: The model to train, evaluate or use for predictions.
            args: The arguments to tweak for training.
            data_collator: The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`.
            train_dataset: The dataset to use for training.
            eval_dataset: The dataset to use for evaluation.
            tokenizer: The tokenizer used to preprocess the data.
            model_init: A function that instantiates the model to be used.
            compute_metrics: A function that will be called at the end of each evaluation phase.
            callbacks: A list of callbacks to customize the training loop.
            optimizers: A tuple containing the optimizer and the scheduler to use.
            training_mode (TrainingMode): The training mode, default to ALIGNMENT.
            **kwargs: Additional keyword arguments to pass to the Trainer.
        """
        # # Initialize the accelerator
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs
        )
        self.training_mode = training_mode
        # self.on_the_fly_embedding = on_the_fly_embedding
        self.enable_pytorch_profiling = os.environ.get('ENABLE_PYTORCH_PROFILER', None) == '1' and \
            self.state.is_world_process_zero
        self.pytorch_profiler_config = pytorch_profiler_config if pytorch_profiler_config is not None else {}
        self.model_accepts_loss_kwargs = False

    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()

        if os.environ.get('ENABLE_BENCHY', None) == '1':
            train_dataloader = BenchmarkGenericIteratorWrapper(
                train_dataloader, self.args.per_device_train_batch_size)

        return train_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation for multimodal inputs.

        Args:
            model: The model to compute the loss.
            inputs: The inputs from the DataLoader.
            return_outputs: Whether or not to return the outputs.

        Returns:
            The loss or (loss, outputs) if return_outputs is True.
        """

        # Prepare model inputs
        model_inputs = {
            'attention_mask': inputs.get('attention_mask', None),
            'labels': inputs['labels'],
            'position_ids': inputs['position_ids'],
        }

        model_inputs['input_ids'] = inputs['input_ids']
        model_inputs['processed_multimodal_inputs'] = inputs['processed_multimodal_inputs']

        outputs = model(**model_inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


    def train(self, *args, **kwargs):
        """
        Custom training loop that sets the model in the correct training mode before training.

        Args:
            *args: Positional arguments passed to the Trainer's train method.
            **kwargs: Keyword arguments passed to the Trainer's train method.

        Returns:
            A `TrainOutput` object with training information.
        """
        self.model.train()

        # Set the model in the correct training mode
        if self.training_mode == TrainingMode.ALIGNMENT:
            self.model.freeze_for_alignment()
        elif self.training_mode == TrainingMode.LM_ONLY:
            self.model.freeze_for_lm()
        elif self.training_mode == TrainingMode.END2END:
            self.model.freeze_for_end2end()
        elif self.training_mode == TrainingMode.FULL:
            self.model.unfreeze()
        else:
            raise ValueError(f"Unknown training mode {self.training_mode}")

        # Pytorch profiler (avoid with NGC Pytorch 24.11-25.01)
        if self.enable_pytorch_profiling:
            from torch.profiler import profile, ProfilerActivity

            wait_steps = int(
                self.pytorch_profiler_config.get('wait_steps', 25))
            warmup_steps = int(
                self.pytorch_profiler_config.get('warmup_steps', 25))
            active_steps = int(
                self.pytorch_profiler_config.get('active_steps', 1))

            if self.args.max_steps < wait_steps + warmup_steps + active_steps:
                warnings.warn(
                    f"Profiler will not run: max_steps ({self.args.max_steps}) should be greater than wait_steps ({wait_steps}) + warmup_steps ({warmup_steps}) + active_steps ({active_steps})")

            print(
                f"Enabling Pytorch profiling (wait={wait_steps}, warmup={warmup_steps}, active={active_steps} steps)")

            def trace_handler(p):
                trace_filename = f"logs/R-{os.environ.get('SLURM_JOB_NAME')}.{os.environ.get('SLURM_JOBID')}_"\
                    f"pttrace_s{str(wait_steps+warmup_steps)}_{str(wait_steps+warmup_steps+active_steps)}"\
                    f"_r{os.environ.get('SLURM_PROCID')}.json"
                p.export_chrome_trace(trace_filename)
                print(f"Exported Pytorch profiler trace to {trace_filename}")

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                # record_shapes=True,
                with_stack=True,
                # profile_memory=True,
                # with_flops=True,
                schedule=torch.profiler.schedule(
                    wait=wait_steps, warmup=warmup_steps, active=active_steps),
                on_trace_ready=trace_handler,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    verbose=True)
            ) as profiler:
                self.profiler = profiler
                return super().train(*args, **kwargs)

        else:
            return super().train(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        if self.enable_pytorch_profiling:
            from torch.profiler import record_function
            with record_function("training_step"):
                ret = super().training_step(*args, **kwargs)

            self.profiler.step()
            return ret
        else:
            return super().training_step(*args, **kwargs)
