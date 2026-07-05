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
    """
    Custom HuggingFace Trainer subclass for multimodal training.
    Handles gradient norm logging and optional debug tensor saving.
    """

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

        # ── DEBUG state (used by compute_loss when DEBUG_OUT_DIR is set) ──────
        self._debug_step_done   = False
        self._debug_captures    = {"enc": None, "proj": None}
        self._hooks_registered  = False
        self._debug_out         = os.environ.get("DEBUG_OUT_DIR", "")
        if self._debug_out:
            self._debug_out = os.path.join(self._debug_out, "multimeditron")
            os.makedirs(self._debug_out, exist_ok=True)
            print(f"[DEBUG] MultiMeditron: debug mode enabled, saving to {self._debug_out}")
        else:
            self._hooks_registered = True   # nothing to register
        # ─────────────────────────────────────────────────────────────────────

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a custom optimizer if custom_lr is given for different modality parts.
        """
        if self.optimizer is None:
            if not self.custom_lr:
                # Default behavior
                return super().create_optimizer()
                
            # We construct groups based on component logic
            # "projector" will get lr_mp
            # "embedder" (vision) will get lr_vision
            # the rest will get the default lr
            
            decay_parameters = self.get_decay_parameter_names(self.model)
            
            optimizer_grouped_parameters = []
            
            default_lr = self.args.learning_rate
            lr_projector = self.custom_lr.get("projector", default_lr)
            lr_vision = self.custom_lr.get("vision", default_lr)
            
            params_projector_decay = []
            params_projector_nodecay = []
            params_vision_decay = []
            params_vision_nodecay = []
            params_default_decay = []
            params_default_nodecay = []
            
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                # Determine which group this parameter belongs to
                if "projector" in n:
                    if n in decay_parameters:
                        params_projector_decay.append(p)
                    else:
                        params_projector_nodecay.append(p)
                elif "feature_extractor" in n or "vision" in n:
                    if n in decay_parameters:
                        params_vision_decay.append(p)
                    else:
                        params_vision_nodecay.append(p)
                else: # Default (LLM + others)
                    if n in decay_parameters:
                        params_default_decay.append(p)
                    else:
                        params_default_nodecay.append(p)
            
            optimizer_grouped_parameters.extend([
                {"params": params_projector_decay, "weight_decay": self.args.weight_decay, "lr": lr_projector},
                {"params": params_projector_nodecay, "weight_decay": 0.0, "lr": lr_projector},
                {"params": params_vision_decay, "weight_decay": self.args.weight_decay, "lr": lr_vision},
                {"params": params_vision_nodecay, "weight_decay": 0.0, "lr": lr_vision},
                {"params": params_default_decay, "weight_decay": self.args.weight_decay, "lr": default_lr},
                {"params": params_default_nodecay, "weight_decay": 0.0, "lr": default_lr},
            ])
            
            # Filter empty groups
            optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            # Remove lr from optimizer_kwargs as we already assigned it to parameter groups
            if "lr" in optimizer_kwargs:
                del optimizer_kwargs["lr"]

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_train_dataloader(self):
        train_dataloader = super().get_train_dataloader()

        if os.environ.get('ENABLE_BENCHY', None) == '1':
            train_dataloader = BenchmarkGenericIteratorWrapper(
                train_dataloader, self.args.per_device_train_batch_size)

        return train_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation for multimodal inputs.
        """
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

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Inject the captured per-component gradient norms into WandB logs.
        """
        if hasattr(self, "_latest_grad_norms"):
            logs.update(self._latest_grad_norms)

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def train(self, *args, **kwargs):
        """
        Custom training loop that sets the model in the correct training mode before training.
        """
        # ── DEBUG: run fixed shared batch through the real pipeline ──────────
        _debug_batch_file = os.environ.get("DEBUG_BATCH_FILE", "")
        if self._debug_out and _debug_batch_file:
            import pickle, io, numpy as np
            from PIL import Image as _PIL

            print(f"[DEBUG] MultiMeditron: running fixed-batch debug, saving to {self._debug_out}")
            with open(_debug_batch_file, "rb") as _f:
                _debug_samples = pickle.load(_f)

            # Register forward hooks on the vision encoder and projector
            _actual = self.model.module if hasattr(self.model, 'module') else self.model
            _modality = list(_actual.modalities_by_type.values())[0]
            _enc_cap, _proj_cap = {}, {}
            def _hook_enc(module, inp, out):
                h = out.last_hidden_state
                if hasattr(_modality.feature_extractor.vision_model, 'post_layernorm'):
                    h = _modality.feature_extractor.vision_model.post_layernorm(h)
                _enc_cap['out'] = h.detach().float().cpu()
            def _hook_proj(module, inp, out):
                _proj_cap['out'] = out.detach().float().cpu()
            _modality.feature_extractor.vision_model.register_forward_hook(_hook_enc)
            _modality.projector.register_forward_hook(_hook_proj)

            _all_ids, _all_lbl, _all_enc, _all_proj, _all_logits, _all_loss = [], [], [], [], [], []

            self.model.eval()
            for _s in _debug_samples:
                _img = _PIL.open(io.BytesIO(_s["image_bytes"])).convert("RGB")
                _conv = []
                _first = True
                for _msg in _s["conversations"]:
                    _role = _msg["role"]
                    _content = _msg["content"]
                    # Strip any stray image tokens already in the content exactly like nanoVLM
                    _content = _content.replace(self.data_collator.attachment_token, "")
                    if _role == "user" and _first:
                        _content = self.data_collator.attachment_token + _content
                        _first = False
                    _conv.append({"role": _role, "content": _content})

                # Build an Arrow-style sample dict and run it through the real collator
                _raw = {
                    "conversations": _conv,
                    "modalities": [{"type": "image", "value": {"bytes": _s["image_bytes"]}}]
                }
                _batch = self.data_collator([_raw])
                _device = next(self.model.parameters()).device
                _batch_d = {k: v.to(_device) if isinstance(v, torch.Tensor) else v for k, v in _batch.items()}

                # ── TOKENIZER DEBUG: print first sample only ──────────────────
                if len(_all_ids) == 0:
                    _tok = self.data_collator.tokenizer
                    _ids_cpu = _batch_d['input_ids'][0].cpu()
                    _img_tok_id = _tok.convert_tokens_to_ids(self.data_collator.attachment_token)
                    _glob_tok   = self.data_collator.chat_template.special_tokens.get('global_image')
                    _glob_tok_id = _tok.convert_tokens_to_ids(_glob_tok) if _glob_tok else None
                    _n_img  = (_ids_cpu == _img_tok_id).sum().item()
                    _n_glob = (_ids_cpu == _glob_tok_id).sum().item() if _glob_tok_id is not None else 0
                    print("[DEBUG][MultiMeditron] === SAMPLE 0 TOKENIZER REPORT ===")
                    print(f"[DEBUG][MultiMeditron] vocab_size         = {len(_tok)}")
                    print(f"[DEBUG][MultiMeditron] image_token        = '{self.data_collator.attachment_token}'  id={_img_tok_id}")
                    print(f"[DEBUG][MultiMeditron] global_image_token = '{_glob_tok}'  id={_glob_tok_id}")
                    print(f"[DEBUG][MultiMeditron] seq_len            = {len(_ids_cpu)}")
                    print(f"[DEBUG][MultiMeditron] image tokens count = {_n_img}")
                    print(f"[DEBUG][MultiMeditron] global_img  count  = {_n_glob}")
                    print(f"[DEBUG][MultiMeditron] first 20 token IDs = {_ids_cpu[:20].tolist()}")
                    _chat_str = _tok.apply_chat_template(
                        _conv, tokenize=False, add_generation_prompt=False
                    )
                    print(f"[DEBUG][MultiMeditron] chat_template_str (first 300 chars):")
                    print(repr(_chat_str[:300]))
                    print(f"[DEBUG][MultiMeditron] first 20 tokens decoded:")
                    for _ti, _tid in enumerate(_ids_cpu[:20].tolist()):
                        print(f"  [{_ti:02d}] id={_tid:6d}  -> {repr(_tok.decode([_tid]))}")
                    print("[DEBUG][MultiMeditron] =======================================")
                # ─────────────────────────────────────────────────────────────

                with torch.no_grad():
                    _out = self.model(
                        input_ids=_batch_d['input_ids'],
                        labels=_batch_d['labels'],
                        attention_mask=_batch_d['attention_mask'],
                        position_ids=_batch_d['position_ids'],
                        processed_multimodal_inputs=_batch_d['processed_multimodal_inputs'],
                    )

                _all_ids.append(_batch_d['input_ids'][0].cpu().numpy())
                _all_lbl.append(_batch_d['labels'][0].cpu().numpy())
                _all_enc.append(_enc_cap['out'][0].numpy()  if _enc_cap  else np.array([]))
                _all_proj.append(_proj_cap['out'][0].numpy() if _proj_cap else np.array([]))
                _all_logits.append(_out.logits[0].detach().float().cpu().numpy())
                _all_loss.append(_out.loss.item())

            def _save_jagged(path, lst):
                arr = np.empty(len(lst), dtype=object)
                arr[:] = lst
                np.save(path, arr)

            _save_jagged(os.path.join(self._debug_out, "tokenizer_input_ids.npy"), _all_ids)
            _save_jagged(os.path.join(self._debug_out, "tokenizer_labels.npy"),    _all_lbl)
            _save_jagged(os.path.join(self._debug_out, "encoder_features.npy"),    _all_enc)
            _save_jagged(os.path.join(self._debug_out, "projector_output.npy"),    _all_proj)
            _save_jagged(os.path.join(self._debug_out, "lm_logits.npy"),           _all_logits)
            np.save(os.path.join(self._debug_out, "loss.npy"),                np.array(_all_loss))
            print(f"[DEBUG] MultiMeditron done. avg_loss={np.mean(_all_loss):.4f}  "
                  f"enc_std={np.mean([x.std() for x in _all_enc if x.size > 0]):.4f}  "
                  f"proj_std={np.mean([x.std() for x in _all_proj if x.size > 0]):.4f}")
        # ─────────────────────────────────────────────────────────────────────

        self.model.train()

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

        if self.enable_pytorch_profiling:
            from torch.profiler import profile, ProfilerActivity
            wait_steps = int(self.pytorch_profiler_config.get('wait_steps', 25))
            warmup_steps = int(self.pytorch_profiler_config.get('warmup_steps', 25))
            active_steps = int(self.pytorch_profiler_config.get('active_steps', 1))

            if self.args.max_steps < wait_steps + warmup_steps + active_steps:
                warnings.warn(f"Profiler will not run: max_steps ({self.args.max_steps}) should be greater than wait_steps ({wait_steps}) + warmup_steps ({warmup_steps}) + active_steps ({active_steps})")

            print(f"Enabling Pytorch profiling (wait={wait_steps}, warmup={warmup_steps}, active={active_steps} steps)")

            def trace_handler(p):
                trace_filename = f"logs/R-{os.environ.get('SLURM_JOB_NAME')}.{os.environ.get('SLURM_JOBID')}_pttrace_s{str(wait_steps+warmup_steps)}_{str(wait_steps+warmup_steps+active_steps)}_r{os.environ.get('SLURM_PROCID')}.json"
                p.export_chrome_trace(trace_filename)
                print(f"Exported Pytorch profiler trace to {trace_filename}")

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps),
                on_trace_ready=trace_handler,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
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
        else:
            ret = super().training_step(*args, **kwargs)

        # Capture gradients right AFTER backward() but BEFORE zero_grad()
        if self.model is not None:
            import math
            if not hasattr(self, "_latest_grad_norms"):
                self._latest_grad_norms = {}
                
            grad_norms = {"projector": [], "vision": [], "llm": []}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.detach().float().norm(2).item()
                    if "projector" in name or "MP" in name:
                        grad_norms["projector"].append(norm)
                    elif "feature_extractor" in name or "vision" in name:
                        grad_norms["vision"].append(norm)
                    else:
                        grad_norms["llm"].append(norm)

            for component, norms in grad_norms.items():
                if norms:
                    self._latest_grad_norms[f"grad_norm_{component}"] = math.sqrt(sum(n ** 2 for n in norms))

        return ret

