import os
import threading
import importlib
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
from enum import IntEnum
from transformers import Trainer
from accelerate import Accelerator
from typing import Optional, Any, Dict, Union
import warnings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Packed-sequence attention fix
# ---------------------------------------------------------------------------
# HF's FA2 LLaMA derives cu_seqlens from attention_mask.sum(-1), which treats
# an entire packed bin as one causal sequence → cross-sample attention leakage.
# We fix this by replacing _get_unpad_data with a version that uses our
# pre-computed cu_seqlens (one per bin, encoding sub-sequence boundaries).
# The replacement is activated only when _PACKING_CONTEXT.cu_seqlens is set.
# ---------------------------------------------------------------------------

_PACKING_CONTEXT = threading.local()


def _patched_get_unpad_data(attention_mask: torch.Tensor):
    """
    Drop-in replacement for transformers._get_unpad_data.
    When called during a packed forward pass (cu_seqlens stored on
    _PACKING_CONTEXT), returns sub-sequence-level boundaries so that
    flash_attn_varlen_func applies causal masking within each sub-sequence
    instead of across the entire packed bin.
    """
    # indices: positions of real (non-padding) tokens — always correct as-is
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    cu_seqlens_list = getattr(_PACKING_CONTEXT, "cu_seqlens", None)

    if cu_seqlens_list is None:
        # Non-packed path: standard HF behaviour
        max_seqlen = int(seqlens_in_batch.max().item())
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        return indices, cu_seqlens, max_seqlen

    # Packed path: flatten sub-sequence lengths across ALL bins.
    # cu_seqlens_list[b] = int32 tensor [0, s0, s0+s1, ..., total_real, max_len]
    # Real intervals: [cu[i], cu[i+1]) for i in 0..(len-3).
    # The last interval [total_real, max_len] is padding — skip it.
    flat_seqlens: List[int] = []
    for b_cu in cu_seqlens_list:
        b_list = b_cu.tolist()
        for i in range(len(b_list) - 2):          # -2: skip padding sentinel
            slen = int(b_list[i + 1]) - int(b_list[i])
            if slen > 0:
                flat_seqlens.append(slen)

    if not flat_seqlens:
        # Fallback (should not happen in normal operation)
        max_seqlen = int(seqlens_in_batch.max().item())
        cu_seqlens = F.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
        )
        return indices, cu_seqlens, max_seqlen

    device = attention_mask.device
    seqlens_t = torch.tensor(flat_seqlens, dtype=torch.int32, device=device)
    max_seqlen = int(seqlens_t.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_t, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _apply_packing_patch() -> List[str]:
    """
    Patch _get_unpad_data in all locations where HF transformers imports it.
    Returns the list of module names that were successfully patched.
    Robust to different transformers versions (4.34 – 4.47+).
    """
    patched: List[str] = []
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.modeling_flash_attention_utils",
        "transformers.models.mistral.modeling_mistral",
    ]
    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "_get_unpad_data"):
                mod._get_unpad_data = _patched_get_unpad_data
                patched.append(mod_name)
        except ImportError:
            pass
    return patched


_PACKING_PATCH_LOCATIONS: List[str] = []
_packing_patch_initialized = False


def init_packing_patch() -> List[str]:
    """Apply the packed-sequence FA2 attention patch (idempotent).

    Safe to call any number of times — the underlying monkey-patch is applied at
    most once. Called explicitly at training startup (see ``cli/train.py``) rather
    than at import, so importing the trainer has no side effect. Eval does not use
    packing and therefore never needs the patch.

    Returns:
        The list of transformers module names that were patched.
    """
    global _packing_patch_initialized, _PACKING_PATCH_LOCATIONS
    if _packing_patch_initialized:
        return _PACKING_PATCH_LOCATIONS

    import transformers
    _PACKING_PATCH_LOCATIONS = _apply_packing_patch()
    _packing_patch_initialized = True
    if _PACKING_PATCH_LOCATIONS:
        logger.info("Packed-sequence attention patch applied (transformers %s) in: %s",
                    transformers.__version__, ", ".join(_PACKING_PATCH_LOCATIONS))
    else:
        logger.warning(
            "Could not apply packed-sequence attention patch on transformers %s: "
            "_get_unpad_data not found in known modules. Cross-sample attention "
            "leakage may occur with pack_sequences=True — verify the transformers "
            "version exposes _get_unpad_data and update _apply_packing_patch() if not.",
            transformers.__version__,
        )
    return _PACKING_PATCH_LOCATIONS


def setup_benchy():
    """Import benchy's iterator wrapper, only when ``ENABLE_BENCHY=1``.

    Keeps the optional ``benchy`` dependency out of module import.

    Returns:
        The ``BenchmarkGenericIteratorWrapper`` class, or None if disabled.
    """
    if os.environ.get('ENABLE_BENCHY') == '1':
        from benchy.torch import BenchmarkGenericIteratorWrapper
        return BenchmarkGenericIteratorWrapper
    return None


class TrainingMode(IntEnum):
    """Enumeration of supported multimodal training modes."""

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
        model_init=None,
        compute_metrics=None,
        # on_the_fly_embedding: bool = True,
        callbacks=None,
        optimizers=(None, None),
        training_mode: TrainingMode = TrainingMode.ALIGNMENT,
        pytorch_profiler_config=None,
        custom_lr: Optional[Dict[str, float]] = None,
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
            model_init: A function that instantiates the model to be used.
            compute_metrics: A function that will be called at the end of each evaluation phase.
            callbacks: A list of callbacks to customize the training loop.
            optimizers: A tuple containing the optimizer and the scheduler to use.
            training_mode (TrainingMode): The training mode, default to ALIGNMENT.
            custom_lr (dict, optional): Per-component learning rates. Keys: ``projector``, ``vision``.
                If omitted, all parameters use ``training_args.learning_rate``.
            **kwargs: Additional keyword arguments to pass to the Trainer.
        """
        # # Initialize the accelerator
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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
        self.custom_lr = custom_lr or {}

    def create_optimizer(self):
        """Setup the optimizer, with optional per-component learning rates.

        If ``custom_lr`` is provided, the parameters are split into three groups:
        ``projector`` (keys containing ``projector``), ``vision`` (keys containing
        ``feature_extractor`` or ``vision``), and everything else (LLM). Each group
        gets its own learning rate while respecting weight-decay settings.
        """
        if self.optimizer is not None or not self.custom_lr:
            return super().create_optimizer()

        default_lr = self.args.learning_rate
        lr_projector = self.custom_lr.get("projector", default_lr)
        lr_vision = self.custom_lr.get("vision", default_lr)

        decay_parameters = self.get_decay_parameter_names(self.model)

        params_projector_decay, params_projector_nodecay = [], []
        params_vision_decay, params_vision_nodecay = [], []
        params_default_decay, params_default_nodecay = [], []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "projector" in n:
                (params_projector_decay if n in decay_parameters else params_projector_nodecay).append(p)
            elif "feature_extractor" in n or "vision" in n:
                (params_vision_decay if n in decay_parameters else params_vision_nodecay).append(p)
            else:
                (params_default_decay if n in decay_parameters else params_default_nodecay).append(p)

        optimizer_grouped_parameters = [
            {"params": params_projector_decay,  "weight_decay": self.args.weight_decay, "lr": lr_projector},
            {"params": params_projector_nodecay, "weight_decay": 0.0,                   "lr": lr_projector},
            {"params": params_vision_decay,      "weight_decay": self.args.weight_decay, "lr": lr_vision},
            {"params": params_vision_nodecay,    "weight_decay": 0.0,                   "lr": lr_vision},
            {"params": params_default_decay,     "weight_decay": self.args.weight_decay, "lr": default_lr},
            {"params": params_default_nodecay,   "weight_decay": 0.0,                   "lr": default_lr},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)  # lr is already per-group
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_train_dataloader(self):
        """Return the training DataLoader, optionally wrapped with a Benchy benchmark iterator.

        Returns:
            DataLoader: The training dataloader, optionally wrapped for throughput benchmarking
                when the ENABLE_BENCHY environment variable is set to '1'.
        """
        train_dataloader = super().get_train_dataloader()

        benchy_wrapper = setup_benchy()
        if benchy_wrapper is not None:
            train_dataloader = benchy_wrapper(
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

        # Extract cu_seqlens produced by the packing collator (not a model input).
        # When present, _patched_get_unpad_data will use them to give FA2 the correct
        # sub-sequence boundaries so attention does not leak across packed sequences.
        cu_seqlens = inputs.get("cu_seqlens", None)

        # Prepare model inputs
        model_inputs = {
            'attention_mask': inputs.get('attention_mask', None),
            'labels': inputs['labels'],
            'position_ids': inputs['position_ids'],
        }

        model_inputs['input_ids'] = inputs['input_ids']
        model_inputs['processed_multimodal_inputs'] = inputs['processed_multimodal_inputs']

        _PACKING_CONTEXT.cu_seqlens = cu_seqlens
        try:
            outputs = model(**model_inputs)
        finally:
            _PACKING_CONTEXT.cu_seqlens = None

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
        """Execute a single training step, optionally recording a PyTorch profiler trace.

        Args:
            *args: Positional arguments passed to the parent Trainer's training_step.
            **kwargs: Keyword arguments passed to the parent Trainer's training_step.

        Returns:
            torch.Tensor: The training loss for this step.
        """
        if self.enable_pytorch_profiling:
            from torch.profiler import record_function
            with record_function("training_step"):
                ret = super().training_step(*args, **kwargs)

            self.profiler.step()
            return ret
        else:
            return super().training_step(*args, **kwargs)
