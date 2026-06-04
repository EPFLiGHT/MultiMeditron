.. _config-ref-label:

Configuration Reference
=======================


.. code-block:: yaml

    base_llm: # (str) Path to LLM model (can be a local model or a model stored on huggingface)
    base_model: # (str) Path to trained model. If empty, the LLM model will be initialized to the weights of base_llm, the modality embedders are initialized to their default values and projections are initialized randomly
    attachment_token: # (str) Attachment placeholder in the prompts. Default to <|reserved_special_token_0|>
    tokenizer_type: # (str) The type of tokenizer that should be used, depends on the model (supported values are llama, apertus and qwen3)
    token_size: # (int) Dimension of the embedding of a token for the LLM
    
    # Truncation settings
    truncation: # (Optional[boolean]) Whether to truncate the input or not, default to false
    max_sequence_length: # (Optional[int]) The maximum sequence length if truncation is enabled

    # Reload from checkpoint
    resume_from_checkpoint: # (Optional[bool]) Whether to resume training from checkpoint, default to false. If set to true, the training will resume from the checkpoint in base_model
    wandb_run_id: # (Optional[str]) The wandb run id to resume from if resume_from_checkpoint is true


    modalities:
        config: # (Dict[str, str]) Configuration passed to the modality
            model_type: # (str) Type of the modality used (e.g. meditron_clip or moe_meditron_clip for instance)
            # The other parameters in config are passed in the modality configuration

    training_mode: # (str) Either ALIGNMENT, END2END or FULL. If ALIGNMENT, this will train the projection layer while freezing every other weights. If END2END, this will train the LLM+Projection while freezing every other weights. If FULL, this will train all the model at the same time

    loaders:
        - loader_type: # (str) Type of the loader. Supported values are: raw-image (for image bytes/PIL images), fs-image (for image paths on the filesystem, not recommended)
          modality_type: # (str) Type of the modality that this loader corresponds to (e.g. image)

    datasets: # List of datasets to use for finetuning. Each dataset must follow the format described in the README.md
      - packed_path: # (str) Path to the 1st dataset
      - packed_path: # (str) Path to the 2nd dataset

    training_args: # Huggingface training arguments. Check the following documentation for more informations: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


MoE Configuration
-----------------

When using a Mixture-of-Experts (MoE) vision encoder, the ``modalities`` section requires additional fields.

``modalities.config.model_type``
   Set to ``moe_meditron_clip`` (or ``moe_meditron_clip_pep`` for PEP-gated variants).

``modalities.config.expert_clip_names``
   A list of paths to the expert CLIP model checkpoints. Each entry corresponds to one domain expert (e.g. CT, MRI, X-ray).

``modalities.config.gating_path``
   Path to the trained gating network that routes images to the appropriate experts.

``modalities.config.fusion_method``
   How expert outputs are combined. Supported values: ``cross_attn`` (attention-based fusion) or ``average`` (simple averaging).

``modalities.config.expert_projection``
   Projection strategy for expert embeddings. Supported values: ``per_expert`` (one projection per expert) or ``shared`` (single shared projection).

``modalities.config.image_processor``
   Path to the shared image processor used by all experts (e.g. a CLIP ViT-B/32 processor).

``modalities.config.top_k_experts``
   Number of experts to activate per image (optional, defaults to all).

Below is a complete example YAML snippet for an MoE configuration:

.. code-block:: yaml

    modalities:
      - model_type: moe_meditron_clip_pep
        image_processor: /path/to/clip-vit-base-patch32
        hidden_size: 4096
        expert_clip_names:
          - /path/to/expert-ct
          - /path/to/expert-mri
          - /path/to/expert-ultrasound
          - /path/to/expert-xray
          - /path/to/expert-generalist
        gating_path: /path/to/gating-network
        fusion_method: cross_attn
        top_k_experts: 3


