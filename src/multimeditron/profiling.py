import torch
from transformers import TrainerCallback


class NvtxAnnotationCallback(TrainerCallback):
    """"
    Adding NVTX annotations for profiling with Nsight Systems.
    """

    def __init__(self, global_step_start=100, global_step_stop=120):
        """Initialize the profiling callback.

        Args:
            global_step_start (int): Global step at which to start CUDA profiling.
                Defaults to 100.
            global_step_stop (int): Global step at which to stop CUDA profiling.
                Defaults to 120.
        """
        self.global_step_start = global_step_start
        self.global_step_stop = global_step_stop

    # other kwargs of callbacks: model, tokenizer, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    def on_init_end(self, args, state, control, **kwargs):
        pass

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, **kwargs):
        """Start CUDA profiling at the configured step and push an NVTX range for the training step."""
        if state.global_step == self.global_step_start and state.is_world_process_zero:
            torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push(f"step {state.global_step}")

    def on_prepare_inputs_begin(self, args, state, control, **kwargs):
        """Push an NVTX range marking the beginning of data copy to device."""
        torch.cuda.nvtx.range_push(f"data copy in {state.global_step}")

    def on_prepare_inputs_end(self, args, state, control, **kwargs):
        """Pop the NVTX range for the data copy phase."""
        torch.cuda.nvtx.range_pop() # copy in

    def on_forward_begin(self, args, state, control, **kwargs):
        """Push an NVTX range marking the beginning of the forward pass."""
        torch.cuda.nvtx.range_push(f"forward")

    def on_forward_end(self, args, state, control, **kwargs):
        """Pop the NVTX range for the forward pass."""
        torch.cuda.nvtx.range_pop() #forward

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Push an NVTX range marking the beginning of the optimizer step."""
        torch.cuda.nvtx.range_push(f"optimizer")

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Pop the NVTX range for the optimizer step."""
        torch.cuda.nvtx.range_pop() # optimizer

    def on_step_end(self, args, state, control, **kwargs):
        """Pop the NVTX range for the training step and stop CUDA profiling at the configured step."""
        torch.cuda.nvtx.range_pop() # step
        if state.global_step == self.global_step_stop and state.is_world_process_zero:
            torch.cuda.profiler.stop()

    def on_substep_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        """Stop CUDA profiling at the end of the target epoch."""
        if state.epoch == self.epoch_to_profile:
            torch.cuda.profiler.stop()

    def on_train_end(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs, **kwargs):
        pass

    def on_evaluate(self, args, state, control, output_metrics, **kwargs):
        pass

    def on_predict(self, args, state, control, output_metrics, **kwargs):
        pass

    def on_prediction_step(self, args, state, control, **kwargs):
        pass
