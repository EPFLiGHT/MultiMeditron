import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple, Any, Dict
from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig, AutoProcessor, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

from multimeditron.model.modalities import BaseModalityProcessor, AutoModality, BaseModalityConfig, BaseModality
from multimeditron.utils import get_torch_dtype
import logging
import json
import tempfile

logger = logging.getLogger(__name__)

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
        attachment_token_idx: int = 1,
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
            attachment_token_idx (int): Index of the attachment token in the vocabulary. Defaults to 1.
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
        self.attachment_token_idx = attachment_token_idx
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
        self.modalities = nn.ModuleList()

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
            self.modalities.append(modality)

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
        for modality in self.modalities:
            modality.freeze_modality_only()
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
        for modality in self.modalities:
            modality.freeze_all()
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
        for modality in self.modalities:
            modality.freeze_modality_only()
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
        for modality in self.modalities:
            modality.unfreeze_all()
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

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
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

        # Compute the projection and scatter into embedded token sequence
        for modality_name, processed_modality_stack in processed_multimodal_inputs['stacked'].items():
            modality = self._get_modality_by_name(modality_name)

            embedded_modality_stack = modality(processed_modality_stack)

            embedded_tokens[processed_multimodal_inputs['batch_idx'][modality_name],
                            processed_multimodal_inputs['token_range'][modality_name]] = \
                embedded_modality_stack.view(
                    -1, embedded_modality_stack.shape[-1]).to(embedded_tokens.dtype)

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
        if inputs_embeds is None and multimodal_inputs is None:
            multimodal_inputs = [[]] * input_ids.shape[0]

        if inputs_embeds is None:
            inputs_embeds = self.embed_modalities_with_text(input_ids, processed_multimodal_inputs)

        # Truncate if needed
        if self.config.truncation and self.config.max_sequence_length is not None:
            logger.warning(f"Truncating input to {self.config.max_sequence_length} tokens.")
        
            if inputs_embeds.shape[1] > self.config.max_sequence_length:
                inputs_embeds = inputs_embeds[:, :self.config.max_sequence_length, :]
                if labels is not None:
                    labels = labels[:, :self.config.max_sequence_length]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.config.max_sequence_length]
                if position_ids is not None:
                    position_ids = position_ids[:, :self.config.max_sequence_length]

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

        return torch.cat(generated_tokens).transpose(1, 0)


def bootstrap(config, tokenizer, attachment_token_idx, modalities_config):
    """
    Bootstrap the model and initialize the model as follows:
        - LLM is initialized with the pretrained weights
        - The modalities embedders are initialized with pretrained weights
        - The modalities projector are initialized randomly

    Args:
        

    """
    multimodal_config = MultimodalConfig(
        hidden_size=config["token_size"],
        vocab_size=len(tokenizer),
        attachment_token_idx=attachment_token_idx,
        eos_token_idx=tokenizer.convert_tokens_to_ids(tokenizer.eos_token),
        modalities=modalities_config,
        llm_path=config["base_llm"],
        truncation=config.get("truncation", False),
        max_sequence_length=config.get("max_sequence_length", None),
    )

    model = MultiModalModelForCausalLM(
        multimodal_config, bootstrap=True)
    return model


