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


