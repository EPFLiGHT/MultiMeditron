import time
import logging
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class TimingCallback(TrainerCallback):
    """Tracks and logs mean dataloader wait time per logging interval.

    Dataloader wait is defined as the wall-clock time between the end of
    training step N (``on_step_end``) and the start of step N+1
    (``on_step_begin``).  This captures the time the training loop spends
    blocked waiting for the next prefetched batch to arrive.

    Logs ``dataloader_wait_ms`` (mean over the last ``logging_steps`` steps)
    via the standard Trainer logging mechanism.  Registered as a standard
    ``TrainerCallback`` — add it to the ``callbacks`` list in the trainer
    constructor.  Does not integrate with the training loop automatically.

    Example usage::

        from multimeditron.profiling import TimingCallback
        trainer = MultimodalTrainer(..., callbacks=[TimingCallback()])
    """

    def __init__(self) -> None:
        self._step_end_time: float = 0.0
        self._wait_times_ms: list = []

    def on_step_begin(self, args, state, control, **kwargs) -> None:
        """Record dataloader wait = time since last on_step_end."""
        if self._step_end_time > 0.0:
            wait_ms = (time.perf_counter() - self._step_end_time) * 1000.0
            self._wait_times_ms.append(wait_ms)

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Snapshot the timestamp so the next on_step_begin can compute wait."""
        self._step_end_time = time.perf_counter()

        # Flush accumulated waits every logging_steps
        if self._wait_times_ms and state.global_step % args.logging_steps == 0:
            mean_wait = sum(self._wait_times_ms) / len(self._wait_times_ms)
            logger.info(
                "step %d — mean dataloader wait (last %d steps): %.1f ms",
                state.global_step,
                len(self._wait_times_ms),
                mean_wait,
            )
            # Also surface through Trainer's log dict so it appears in WandB
            control.should_log = True
            self._wait_times_ms.clear()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Flush any remaining timing data at the end of training."""
        self._step_end_time = 0.0
        self._wait_times_ms.clear()


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
