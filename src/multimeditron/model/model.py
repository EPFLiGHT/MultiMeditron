from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Generator, Optional, List, Union, Tuple, Any, Dict, Callable
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, AutoProcessor, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass, field

from multimeditron.model.modalities import BaseModalityProcessor, AutoModality, BaseModalityConfig, BaseModality
from multimeditron.utils import get_torch_dtype
import logging

logger = logging.getLogger(__name__)

@dataclass
class ChatTemplate:
    """
    A generic chat template class to serialize conversation messages
    for different LLM families (LLaMA, Qwen, Apertus, etc.).
    """
    name: str = "custom"

    # Explicit delimiters for each message type
    delimiters: Dict[str, Dict[str, str]] = field(default_factory=dict)
    special_tokens: Dict[str, str] = field(default_factory=dict)

    # ================================================================
    # Built-in templates
    # ================================================================

    @staticmethod
    def from_name(name: str) -> ChatTemplate:
        templates = {
            "llama": ChatTemplate.llama,
            "apertus": ChatTemplate.apertus,
            "qwen3": ChatTemplate.qwen3,
        }
        if name not in templates:
            raise ValueError(f"Unknown chat template name: {name}")
        return templates[name]()

    # -------------------------------
    # LLaMA / Mistral / Vicuna style
    # -------------------------------
    @staticmethod
    def llama() -> ChatTemplate:
        delimiters = {
            "system": {"start": "<|start_header_id|>system<|end_header_id|>", "end": "<|eot_id|>"},
            "user": {"start": "<|start_header_id|>user<|end_header_id|>", "end": "<|eot_id|>"},
            "assistant": {"start": "<|start_header_id|>assistant<|end_header_id|>", "end": "<|eot_id|>"},
        }
        special_tokens = {'image_start': '<|image_start|>', 'image_end': '<|image_end|>'}

        return ChatTemplate(
            name="llama",
            delimiters=delimiters,
            special_tokens=special_tokens
        )

    # -------------------------------
    # Apertus style
    # -------------------------------
    @staticmethod
    def apertus() -> ChatTemplate:
        delimiters = {
            "system": {"start": "<|system_start|>", "end": "<|system_end|>"},
            "developer": {"start": "<|developer_start|>", "end": "<|developer_end|>"},
            "user": {"start": "<|user_start|>", "end": "<|user_end|>"},
            "assistant": {"start": "<|assistant_start|>", "end": "<|assistant_end|>"},
        }
        special_tokens = {'image_start': '<|image_start|>', 'image_end': '<|image_end|>'}

        return ChatTemplate(
            name="apertus",
            delimiters=delimiters,
            special_tokens=special_tokens
        )

    # -------------------------------
    # Qwen 3 / ChatML style
    # -------------------------------
    @staticmethod
    def qwen3() -> ChatTemplate:
        delimiters = {
            "system": {"start": "<|im_start|>system", "end": "<|im_end|>\n"},
            "user": {"start": "<|im_start|>user", "end": "<|im_end|>\n"},
            "assistant": {"start": "<|im_start|>assistant", "end": "<|im_end|>\n"},
        }

        special_tokens = {'global_image': '<|global_image|>'}

        return ChatTemplate(
            name="qwen3",
            delimiters=delimiters,
            special_tokens=special_tokens
        )



@dataclass
class MultimodalConfig(PretrainedConfig):
    """
    Configuration class for a multimodal model that integrates various modalities with a language model.
    """
    model_type = "multimodal"

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        modalities: List[BaseModalityConfig] = [],
        pad_token_idx: int = 0,
        eos_token_idx: int = 0,
        padding_side: str = "left",
        initializer_range: float = 0.02,
        llm_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        truncation: bool = False,
        max_sequence_length: Optional[int] = None,
        dtype="bfloat16",
        **kwargs
    ):
        """
        Initializes the MultimodalConfig.

        Args:
            vocab_size (int, optional): Vocabulary size for the language model. Defaults to None.
            modalities (List[ModalityConfig]): List of modality configurations. Defaults to an empty list.
            pad_token_idx (int): Index of the padding token in the vocabulary. Defaults to 0.
            eos_token_idx (int): Index of the end-of-sequence token in the vocabulary. Defaults to 0.
            padding_side (str): Side for padding sequences ("left" or "right"). Defaults to "left". Choose left for inference, right for training.
            initializer_range (float): Standard deviation for weight initialization. Defaults to 0.02.
            llm_path (str): Path or identifier for the base language model. Defaults to "meta-llama/Llama-3.1-8B-Instruct".
            truncation (bool): Whether to truncate inputs that exceed max_sequence_length. Defaults to False.
            max_sequence_length (int, optional): Maximum sequence length for inputs. Defaults to None.
            dtype (str): Data type for model parameters and computations. Defaults to "bfloat16".
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.modalities = modalities
        self.pad_token_idx = pad_token_idx
        self.eos_token_idx = eos_token_idx
        self.padding_side = padding_side
        self.initializer_range = initializer_range
        self.llm_path = llm_path
        self.dtype = dtype
        self.truncation = truncation
        self.max_sequence_length = max_sequence_length

    def to_dict(self):
        """
        Converts the MultimodalConfig object to a dictionary representation.

        This method extends the parent class's to_dict method by properly handling
        the modalities list, converting each ModalityConfig object to its dictionary
        representation.

        Returns:
            dict: Dictionary containing all configuration parameters, with modalities
                  properly serialized.
        """
        output = super().to_dict()
        output['modalities'] = [modality_config.to_dict()
                                for modality_config in self.modalities]
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Creates a MultimodalConfig instance from a dictionary.

        This classmethod extends the parent class's from_dict method to handle the
        special processing required for modality configurations. It extracts the
        modalities from the configuration dictionary, creates the appropriate
        ModalityConfig objects, and then initializes the MultimodalConfig with these
        processed modalities.

        Args:
            config_dict (dict): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments passed to parent class's from_dict method.
                      Should include 'return_unused_kwargs' which determines the return format.

        Returns:
            Union[MultimodalConfig, Tuple[MultimodalConfig, Dict]]: Either just the config object
            or a tuple of (config, unused_kwargs) if return_unused_kwargs is True.
        """
        modalities_dict_list = config_dict.pop('modalities', [])

        modalities = []
        for modality_dict in modalities_dict_list:
            modalities.append(AutoModality.config_from_dict(modality_dict))

        if kwargs["return_unused_kwargs"]:
            config, kwargs = super().from_dict(config_dict, **kwargs)
            config.modalities = modalities
            return config, kwargs

        config = super().from_dict(config_dict, kwargs)
        config.modalities = modalities
        return config


class MultiModalModelForCausalLM(PreTrainedModel):
    """
    A multimodal model for causal language modeling that integrates various modalities with a language model.

    This model extends PreTrainedModel and is designed to process multiple modalities (such as images,
    audio, etc.) alongside text inputs. It embeds the multimodal inputs into the same embedding space
    as the text tokens and processes them through a shared transformer model.

    The model architecture consists of:

    1. A base language model (like Llama-3)
    2. Multiple modality processors (one for each supported modality)
    3. Projection layers to map modality embeddings to the language model's embedding space

    This enables end-to-end training and inference with multimodal inputs, allowing the model
    to understand and generate text that incorporates information from multiple sources.
    """
    config_class = MultimodalConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: MultimodalConfig,
        bootstrap=False,
    ):
        """
        Initialize a MultiModalModelForCausalLM instance.

        This constructor sets up a multimodal model by integrating a language model with
        various modality processors. It creates the base language model, configures it with
        the appropriate vocabulary size, and initializes all required modality processors
        based on the configuration.

        Args:
            config (MultimodalConfig): The configuration object containing model parameters,
                including modality configurations, vocabulary size, and other settings.
            bootstrap (bool, optional): If True, loads the pretrained model from the path
                specified in config. If False, creates a model from config only. Defaults to False.

        Raises:
            ValueError: If multiple modality configurations of the same type are provided.
        """
        super().__init__(config)

        dtype = get_torch_dtype(config.dtype)

        if bootstrap:
            self.model = AutoModelForCausalLM.from_pretrained(config.llm_path, attn_implementation="flash_attention_2")
        else:
            llm_config = AutoConfig.from_pretrained(
                    config.llm_path,
                    torch_dtype=dtype
                )
            self.model = AutoModelForCausalLM.from_config(
                config=llm_config, attn_implementation="eager")

        self.model.resize_token_embeddings(config.vocab_size, mean_resizing=False)

        # Add the language model to the transformer
        self.modalities_by_type = {}
        self.processors_by_type = {}
        self.modalities_with_projection = nn.ModuleList()

        for modality_config in config.modalities:
            # Retrieve the modality and the number of patches per entry
            modality = AutoModel.from_config(modality_config)
            processor = AutoModality.preprocessor_from_name(modality_config.model_type, modality_config)

            # Ensure there is a single modality per type
            if modality_config.modality_type in self.modalities_by_type:
                raise ValueError(
                    f"Modality type {modality_config.modality_type} has already been registered"
                )

            self.modalities_by_type[modality_config.modality_type] = modality
            self.processors_by_type[modality_config.modality_type] = processor
            self.modalities_with_projection.append(modality)

        # Post init
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize weights for the model modules.

        This method is called during model initialization to set initial values for
        module parameters. Linear layers have their weights initialized from a normal
        distribution and biases set to zero. Embedding layers also have their weights
        initialized from a normal distribution, with special handling for padding indices.

        Args:
            module: The module whose weights should be initialized.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def freeze_for_alignment(self):
        """
        Freezes model parameters for alignment training.

        This method prepares the model for alignment training by:

        1. Freezing only the modality parts of each modality processor (keeping projections trainable)
        2. Freezing the entire language model

        This configuration is useful when aligning modality representations with
        the language model's embedding space while keeping the core LM frozen.
        """
        for modality_with_proj in self.modalities_with_projection:
            modality_with_proj.unfreeze_projection()
            modality_with_proj.freeze_modality_embedder()
        for params in self.model.parameters():
            params.requires_grad = False

    def freeze_for_lm(self):
        """
        Freezes modality parameters for language model fine-tuning.

        This method prepares the model for language model fine-tuning by:

        1. Freezing all modality processors completely (including projections)
        2. Making the language model parameters trainable

        This configuration is useful when you want to fine-tune the language model
        on multimodal inputs while keeping the modality processors fixed.
        """
        for modality_with_proj in self.modalities_with_projection:
            modality_with_proj.freeze_all()
        for params in self.model.parameters():
            params.requires_grad = True

    def freeze_for_end2end(self):
        """
        Freezes partial parameters for end-to-end training.

        This method prepares the model for end-to-end training by:

        1. Freezing only the modality parts of each modality processor (keeping projections trainable)
        2. Making the language model parameters trainable

        This configuration is useful for fine-tuning the language model and modality
        projections together, while keeping the core modality encoders fixed.
        """
        for modality_with_proj in self.modalities_with_projection:
            modality_with_proj.unfreeze_projection()
            modality_with_proj.freeze_modality_embedder()
        for params in self.model.parameters():
            params.requires_grad = True

    def unfreeze(self):
        """
        Unfreezes all model parameters for full training.

        This method makes all parameters of the model trainable by:

        1. Unfreezing all modality processors (both core encoders and projections)
        2. Making the language model parameters trainable

        This configuration enables full end-to-end training of the entire model.
        """
        for modality_with_proj in self.modalities_with_projection:
            modality_with_proj.unfreeze_all()
        for params in self.model.parameters():
            params.requires_grad = True

    def processors(self) -> Dict[str, BaseModalityProcessor]:
        return self.processors_by_type

    def get_model(self):
        return self.model

    def _get_modality_by_name(self, name: str) -> BaseModality:
        if name not in self.modalities_by_type:
            raise KeyError(
                f"No modality registered in the model that can handle modality named: {name}"
            )

        modality = self.modalities_by_type[name]
        if not isinstance(modality, BaseModality):
            raise TypeError(
                f"Registered modality {name} is not of type ModalityWithProjection")

        return modality

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """
        Returns embeddings of the LLM model
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: torch.nn.Embedding):
        """
        Set input embeddings of the LLM model to the given value
        """
        self.model.set_input_embeddings(value)

    def embed_modalities_with_text(self, input_ids: torch.Tensor, processed_multimodal_inputs: List[Dict[str, Any]]):
        """
        Embeds multimodal inputs alongside text tokens in a unified embedding space.

        This method takes text token IDs and processed multimodal inputs, embeds them both,
        and combines them into a single embedding tensor that can be processed by the
        transformer model. It first embeds the text tokens using the model's token embeddings,
        then processes each modality's inputs through their respective modality processors,
        projects them to the language model's hidden dimension, and places them at the
        appropriate positions in the embedding sequence.

        Args:
            input_ids (torch.Tensor): Token IDs for the text input, shape [batch_size, seq_len].
            processed_multimodal_inputs (List[Dict[str, Any]]): Dictionary containing:
                - 'stacked': Dict mapping modality names to tensors of processed inputs
                - 'batch_idx': Dict mapping modality names to batch indices for placement
                - 'token_range': Dict mapping modality names to token indices for placement

        Returns:
            torch.Tensor: Combined embeddings of text and multimodal inputs,
                          shape [batch_size, seq_len, hidden_size].
        """

        embedded_tokens = self.model.get_input_embeddings()(input_ids)

        # Compute the projection and scatter into embedded token sequence.
        # IMPORTANT: We must use out-of-place operations (torch.where) instead of
        # in-place indexed assignment (embedded_tokens[idx] = ...) to preserve the
        # autograd computation graph. In-place writes on tensors that are part of
        # the graph silently break gradient flow to the projector.
        # --- DEBUG: Print Text Embedding Stats ---
        with torch.no_grad():
            if not hasattr(self, "_debug_text_printed"):
                self._debug_text_printed = 0
            if self._debug_text_printed < 3:
                self._debug_text_printed += 1
                print(f"\n--- [DEBUG] Text Embedding Stats ---", flush=True)
                print(f"Mean: {embedded_tokens.mean().item():.6f}", flush=True)
                print(f"Std:  {embedded_tokens.std().item():.6f}", flush=True)
                print("-" * 40, flush=True)
        # -----------------------------------------

        for modality_name, processed_modality_stack in processed_multimodal_inputs['stacked'].items():
            modality = self._get_modality_by_name(modality_name)

            embedded_modality_stack = modality(processed_modality_stack)

            batch_idx = processed_multimodal_inputs['batch_idx'][modality_name]
            token_range = processed_multimodal_inputs['token_range'][modality_name]

            # --- DEBUG: Print SigLIP Output Values ---
            with torch.no_grad():
                _p_count = getattr(self, "_debug_printed", 0)
                if _p_count <= 3:
                    print(f"\n--- [DEBUG] {modality_name} Embedding Values ---", flush=True)
                    print(f"Mean: {embedded_modality_stack.mean().item():.6f}", flush=True)
                    print(f"Std:  {embedded_modality_stack.std().item():.6f}", flush=True)
                    print(f"First 5 values: {embedded_modality_stack.view(-1)[:5].tolist()}", flush=True)
                    print("-" * 40, flush=True)
            # ----------------------------------------

            # Build a boolean mask marking image token positions [batch, seq_len, 1]
            mask = torch.zeros(
                embedded_tokens.shape[:2], dtype=torch.bool, device=embedded_tokens.device
            )
            mask[batch_idx, token_range] = True
            mask = mask.unsqueeze(-1)  # broadcast over hidden_size

            # Build a zeros canvas and place projected visual features into it
            visual_canvas = torch.zeros_like(embedded_tokens)
            visual_canvas[batch_idx, token_range] = \
                embedded_modality_stack.view(-1, embedded_modality_stack.shape[-1]).to(embedded_tokens.dtype)

            # Out-of-place merge: preserves full computation graph for both branches
            embedded_tokens = torch.where(mask, visual_canvas, embedded_tokens)

        return embedded_tokens


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        multimodal_inputs=None,
        processed_multimodal_inputs=None,
        return_dict: Optional[bool] = True,
        cache_position=None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Performs a forward pass through the multimodal model.

        This is the main computation method that processes both text and multimodal inputs.
        It first embeds all inputs (if not already embedded), handles truncation if configured,
        and then passes the combined embeddings through the language model.

        Args:
            input_ids (torch.LongTensor, optional): Token IDs for text input.
                Shape [batch_size, sequence_length].
            inputs_embeds (torch.FloatTensor, optional): Pre-computed input embeddings.
                If provided, input_ids will not be used. Shape [batch_size, sequence_length, hidden_size].
            attention_mask (torch.Tensor, optional): Mask to avoid attention on padding tokens.
                Shape [batch_size, sequence_length].
            position_ids (torch.LongTensor, optional): Indices of positions for positional embeddings.
                Shape [batch_size, sequence_length].
            past_key_values (List[torch.FloatTensor], optional): Cached key/values for faster inference.
            labels (torch.LongTensor, optional): Labels for computing language modeling loss.
                Shape [batch_size, sequence_length].
            use_cache (bool, optional): Whether to return the key/value states for future use.
            multimodal_inputs (Any, optional): Raw multimodal inputs that need processing.
            processed_multimodal_inputs (Dict, optional): Pre-processed multimodal inputs ready for embedding.
            return_dict (bool, optional): Whether to return a dictionary output. Defaults to True.
            cache_position (Any, optional): Position in the cache for retrieval.
            **kwargs: Additional arguments passed to the base model.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]: Model outputs, typically containing:
                - loss (if labels provided)
                - logits (prediction scores for each token)
                - past_key_values (if use_cache=True)
                - hidden_states (if output_hidden_states=True)
                - attentions (if output_attentions=True)
        """
        import sys
        _print_count = getattr(self, "_debug_printed", 0)
        if _print_count < 3 and input_ids is not None:
            self._debug_printed = _print_count + 1
            print("\n" + "="*50, flush=True)
            print(f"MULTIMEDITRON RUNTIME DEBUG (FORWARD PASS {_print_count + 1}/3)", flush=True)
            print(f"Input IDs shape: {input_ids.shape}", flush=True)
            print(f"Input IDs (first 20): {input_ids[0, :20].tolist()}", flush=True)
            if labels is not None:
                print(f"Labels shape: {labels.shape}", flush=True)
                print(f"Valid Labels count (per batch): {(labels != -100).sum(dim=1).tolist()}", flush=True)
            if processed_multimodal_inputs is not None and "stacked" in processed_multimodal_inputs:
                for k, v in processed_multimodal_inputs["stacked"].items():
                    if hasattr(v, "shape"):
                        print(f"Image tensor '{k}' shape: {v.shape}", flush=True)
                    elif isinstance(v, list):
                        if len(v) > 0 and hasattr(v[0], "shape"):
                            print(f"Image list '{k}' length: {len(v)}, first item shape: {v[0].shape}", flush=True)
                        else:
                            print(f"Image list '{k}' length: {len(v)}", flush=True)
                    else:
                        print(f"Image item '{k}' type: {type(v)}", flush=True)
            print("="*50 + "\n", flush=True)
            sys.stdout.flush()

        if inputs_embeds is None and multimodal_inputs is None:
            multimodal_inputs = [[]] * input_ids.shape[0]

        if inputs_embeds is None:
            inputs_embeds = self.embed_modalities_with_text(input_ids, processed_multimodal_inputs)

        # Truncate if needed
        if self.config.truncation and self.config.max_sequence_length is not None:
            if inputs_embeds.shape[1] > self.config.max_sequence_length:
                logger.warning(f"Truncating input to {self.config.max_sequence_length} tokens.")
                inputs_embeds = inputs_embeds[:, :self.config.max_sequence_length, :]
                if labels is not None:
                    labels = labels[:, :self.config.max_sequence_length]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.config.max_sequence_length]
                if position_ids is not None:
                    position_ids = position_ids[:, :self.config.max_sequence_length]

        # --- [DEBUG] INPUT VS LABEL VISUALIZATION ---
        with torch.no_grad():
            if not hasattr(self, "_debug_deep_printed"):
                self._debug_deep_printed = 0
            if self._debug_deep_printed < 2:
                self._debug_deep_printed += 1

                b_idx = 0 # Look at first sample
                if input_ids is not None:
                    ids = input_ids[b_idx]
                    lab = labels[b_idx] if labels is not None else None
                    if lab is not None and ids.shape[0] > lab.shape[0]:
                        ids = ids[:lab.shape[0]]

                    print("\n" + "█"*80, flush=True)
                    print(f"DEEP DEBUG: BATCH SAMPLE {b_idx} ANALYSIS", flush=True)

                    try:
                        from transformers import AutoTokenizer as _DebugTok
                        _tok = _DebugTok.from_pretrained(self.config.llm_path, trust_remote_code=True)

                        # 1. Show the FULL input (What the model sees)
                        full_text = _tok.decode(ids, skip_special_tokens=False)
                        print("\n--- FULL INPUT (MODEL'S VIEW) ---", flush=True)
                        print(full_text[-800:], flush=True)  # Last 800 chars

                        # 2. Show the LEARNED part (What is in the Labels)
                        if lab is not None:
                            valid_mask = lab != -100
                            valid_ids = ids[valid_mask]
                            if len(valid_ids) > 0:
                                learned_text = _tok.decode(valid_ids, skip_special_tokens=False)
                                print("\n--- WHAT THE MODEL IS TRAINING ON (LABELS) ---", flush=True)
                                print(f"LEARNED CONTENT: {learned_text}", flush=True)
                            else:
                                print("\n--- WARNING: LABELS ARE COMPLETELY EMPTY (ALL -100) ---", flush=True)
                    except Exception as _e:
                        print(f"[DEEP DEBUG] Could not decode: {_e}", flush=True)
                        print(f"Input IDs (first 20): {ids[:20].tolist()}", flush=True)
                        if lab is not None:
                            valid_ids = ids[lab != -100]
                            print(f"Label token IDs (first 20): {valid_ids[:20].tolist()}", flush=True)

                    print("█"*80 + "\n", flush=True)
        # ---------------------------------------------

        # Run the transformer model
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

    def inference_generator(
        self,
        batch: Dict[str, Any],
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        **kwargs
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generates text based on the provided batch of inputs.
        This function returns a generator that yields the generated text one token at a time.

        The batch dictionary should contain:
            - 'processed_multimodal_inputs': Processed multimodal inputs.
            - 'input_ids': Input token IDs.
            - 'labels': Optional token IDs for labels.
            - 'attention_mask': Attention mask for the input tokens.
            - 'position_ids': Position IDs for the input tokens.

        This function is particularly useful for inference streaming

        Args:
            batch: A dictionary containing input data, including processed multimodal inputs,
                input ids, and optional labels.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature value for sampling.
            do_sample: Whether to perform sampling during generation.
            kwargs: Additional keyword arguments for the model's inference method.
        """

        input_ids = batch["input_ids"]
        processed_multimodal_inputs = batch["processed_multimodal_inputs"]

        temperature = max(temperature, 1e-6)

        input_ids = input_ids.to(self.model.device)
        device = self.model.device

        # Run the transformer model
        generated_tokens = []
        past_key_values = None
        next_token_embedding = self.embed_modalities_with_text(input_ids, processed_multimodal_inputs)
        finished_mask = torch.zeros(input_ids.shape[0])

        # Get initial attention_mask and position_ids
        attention_mask = batch["attention_mask"].to(device)
        position_ids = batch["position_ids"].to(device)

        seq_length = attention_mask.shape[1]

        with torch.no_grad():
            for i in range(max_new_tokens):
                if i > 0:
                    # For subsequent iterations when using KV cache, we only need position IDs for new token
                    position_ids = (seq_length + i - 1) * torch.ones(
                        (input_ids.shape[0], 1), dtype=torch.long, device=device
                    )

                    # Extend attention mask for the new token
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((input_ids.shape[0], 1), dtype=attention_mask.dtype, device=device)
                    ], dim=-1)

                # Forward pass with embeddings
                outputs = self.model(
                    inputs_embeds=next_token_embedding,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True
                )
                past_key_values = outputs.past_key_values

                # Get the logits and apply softmax to get token probabilities
                # Get logits for the last token
                logits = outputs.logits[:, -1, :].squeeze(1)
                logits = logits / temperature
                softmax = F.softmax(logits, dim=-1)

                if do_sample:
                    next_token_id = []
                    for sample_softmax in softmax:
                        x = torch.multinomial(
                            sample_softmax, num_samples=1)
                        next_token_id.append(x)

                    next_token_id = torch.cat(next_token_id).unsqueeze(0).cpu()
                else:
                    next_token_id = torch.argmax(
                        softmax, dim=-1).unsqueeze(0).cpu()
                yield next_token_id

                # --- [DEBUG] Show model input and first generated token ---
                with torch.no_grad():
                    if i == 0:  # Only on the first generated token
                        if not hasattr(self, "_debug_gen_printed"):
                            self._debug_gen_printed = 0
                        if self._debug_gen_printed < 5:
                            self._debug_gen_printed += 1
                            first_token_id = next_token_id[0].tolist()
                            input_last_ids = input_ids[0, -10:].tolist()
                            print("\n" + "▶"*60, flush=True)
                            print(f"[DEBUG GEN #{self._debug_gen_printed}] LAST 10 INPUT TOKEN IDs: {input_last_ids}", flush=True)
                            print(f"[DEBUG GEN #{self._debug_gen_printed}] FIRST GENERATED TOKEN ID: {first_token_id}", flush=True)
                            print("▶"*60 + "\n", flush=True)
                # -------------------------------------------------------

                for i in range(next_token_id.shape[1]):
                    if finished_mask[i]:
                        next_token_id[0, i] = self.config.eos_token_idx

                # Append the next token to your sequence
                generated_tokens.append(next_token_id)

                finished_mask = torch.logical_or(
                    finished_mask, next_token_id.flatten() == self.config.eos_token_idx)

                if torch.all(finished_mask):
                    break

                # Update the input_embeddings with the embedding of the newly generated token
                next_token_embedding = self.model.get_input_embeddings()(
                    next_token_id.to(input_ids.device)).transpose(1, 0)


    def generate(
        self,
        batch: Dict[str, Any],
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        **kwargs
    ) -> Union[torch.Tensor, CausalLMOutputWithPast]:
        """
        Generates text from multimodal inputs using the model.

        This method implements custom token generation logic for multimodal inputs.
        It processes a batch containing text token IDs and multimodal inputs, then
        performs autoregressive generation of new tokens until either the maximum
        token count is reached or all sequences have generated an end-of-sequence token.

        Args:
            batch (Dict[str, Any]): Dictionary containing the following keys:
                - input_ids: Text token IDs (torch.Tensor)
                - processed_multimodal_inputs: Processed multimodal inputs
                - attention_mask: Attention mask for the input sequence
                - position_ids: Position IDs for the input sequence
            max_new_tokens (int): Maximum number of tokens to generate. Defaults to 512.
            temperature (float): Sampling temperature for controlling randomness in generation.
                Lower values make generation more deterministic. Defaults to 0.1.
            do_sample (bool): Whether to use sampling for generation instead of greedy decoding.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to the underlying generation process.

        Returns:
            torch.Tensor: Generated token IDs, shape [batch_size, sequence_length]
        """
        generated_tokens = []
        generator = self.inference_generator(
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs
        )

        for next_token_id in generator:
            generated_tokens.append(next_token_id)

        result = torch.cat(generated_tokens).transpose(1, 0)

        # --- [DEBUG] Print question and answer in readable form ---
        if not hasattr(self, "_debug_qa_printed"):
            self._debug_qa_printed = 0
        if self._debug_qa_printed < 5:
            self._debug_qa_printed += 1
            try:
                from transformers import AutoTokenizer as _AutoTok
                _tok = _AutoTok.from_pretrained(self.config.llm_path, trust_remote_code=True)
                _input_ids = batch["input_ids"][0]
                _question = _tok.decode(_input_ids, skip_special_tokens=False)
                _answer = _tok.decode(result[0], skip_special_tokens=True)
                print("\n" + "★"*70, flush=True)
                print(f"[QA DEBUG #{self._debug_qa_printed}] QUESTION (last 600 chars):", flush=True)
                print(_question[-600:], flush=True)
                print(f"\n[QA DEBUG #{self._debug_qa_printed}] MODEL ANSWER: '{_answer}'", flush=True)
                print("★"*70 + "\n", flush=True)
            except Exception as _e:
                print(f"[QA DEBUG] Could not decode: {_e}", flush=True)
        # ----------------------------------------------------------

        return result



def bootstrap(config, tokenizer, modalities_config):
    """
    Bootstrap the model and initialize the model as follows:
        - LLM is initialized with the pretrained weights
        - The modalities embedders are initialized with pretrained weights
        - The modalities projector are initialized randomly

    Args:
        config (dict): The configuration dictionary for the multimodal model.
        tokenizer (PreTrainedTokenizerBase): The tokenizer instance to use for tokenization.
        modalities_config (List[BaseModalityConfig]): List of modality configurations.

    Returns:
        MultiModalModelForCausalLM: The initialized multimodal model.
    """


    multimodal_config = MultimodalConfig(
        hidden_size=config["token_size"],
        vocab_size=len(tokenizer),
        eos_token_idx=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        modalities=modalities_config,
        llm_path=config["base_llm"],
        truncation=config.get("truncation", False),
        max_sequence_length=config.get("max_sequence_length", None),
    )

    model = MultiModalModelForCausalLM(
        multimodal_config, bootstrap=True)
    return model



