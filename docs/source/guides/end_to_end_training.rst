.. role:: bash(code)
   :language: bash

End-to-End Training Tutorial
=============================

This tutorial provides a complete step-by-step guide to train a MultiMeditron model from data preparation to inference. We'll walk through the entire pipeline with practical examples.

Overview
--------

The MultiMeditron training pipeline consists of these main stages:

1. **Environment Setup** - Install dependencies and configure your environment
2. **Dataset Preparation** - Format and preprocess your training data
3. **Configuration Setup** - Create training and model configurations
4. **Model Training** - Execute the training process
5. **Inference** - Use your trained model for generation

Prerequisites
-------------

Before starting, ensure you have:

- **Hardware**: NVIDIA GPU with at least 24GB VRAM (for 8B model training)
- **Software**: Python 3.8+, CUDA 11.8+, Docker (optional but recommended)
- **Storage**: At least 100GB free space for models and datasets
- **Access**: HuggingFace account with access to required base models

Step 1: Environment Setup
-------------------------

Option A: Docker Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Pull the pre-built Docker image
    docker pull michelducartier24/multimeditron-git:latest-amd64
    
    # Run the container with GPU access
    docker run --gpus all -it --rm \
        -v $(pwd):/workspace \
        -w /workspace \
        michelducartier24/multimeditron-git:latest-amd64

Option B: Pip Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/EPFLiGHT/MultiMeditron.git
    cd MultiMeditron
    
    # Install PyTorch first (choose your CUDA version)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install MultiMeditron
    pip install -e ".[flash-attn]"

Environment Verification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Check GPU availability
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Check MultiMeditron installation
    python -c "import multimeditron; print('MultiMeditron installed successfully')"

Step 2: Dataset Preparation
---------------------------

MultiMeditron supports two dataset formats:

1. **Arrow/Parquet format** (recommended) - Modalities stored directly in dataset
2. **JSONL format** (deprecated) - Modalities stored on filesystem

Arrow Format Dataset Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **pretraining** datasets:

.. code-block:: python

    # Each sample contains:
    {
        "text": "Let's compare the first image: <|reserved_special_token_0|>, and the second image: <|reserved_special_token_0|>",
        "modalities": [{"type": "image", "value": pil_image_object}, {"type": "image", "value": pil_image_object2}]
    }

For **instruction-tuning** datasets:

.. code-block:: python

    # Each sample contains:
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful medical AI assistant."},
            {"role": "user", "content": "Describe this image: <|reserved_special_token_0|>"},
            {"role": "assistant", "content": "This is a chest X-ray showing..."}
        ],
        "modalities": [{"type": "image", "value": pil_image_object}]
    }

Creating Your Dataset
~~~~~~~~~~~~~~~~~~~~~~

Here's how to create a training dataset from your images:

.. code-block:: python

    from datasets import Dataset
    import json
    from PIL import Image
    
    def create_sample(image_path, text_description, conversations=None):
        # Load image as PIL Image object
        pil_image = Image.open(image_path)
        
        if conversations:  # Instruction-tuning format
            return {
                "conversations": conversations,
                "modalities": [{"type": "image", "value": pil_image}]
            }
        else:  # Pretraining format
            return {
                "text": text_description,
                "modalities": [{"type": "image", "value": pil_image}]
            }
    
    # Example: Create instruction-tuning dataset
    samples = []
    samples.append(create_sample(
        image_path="path/to/your/image1.jpg",
        conversations=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What do you see in this image? <|reserved_special_token_0|>"},
            {"role": "assistant", "content": "I can see a medical image showing..."}
        ]
    ))
    
    # Create and save dataset
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk("my_training_dataset")

Converting JSONL to Arrow
~~~~~~~~~~~~~~~~~~~~~~~~~

If you have JSONL data, convert it using the provided script:

.. code-block:: bash

    # First, create a config for your dataset
    cat > dataset_config.yaml << EOF
    base_llm: meta-llama/Llama-3.1-8B-Instruct
    attachment_token: <|reserved_special_token_0|>
    tokenizer_type: llama
    
    datasets:
      - tokenized_path: ./converted_dataset
        packed_path: ./your_data.jsonl
    EOF
    
    # Convert the dataset
    python merge_inputs.py -c dataset_config.yaml

Step 3: Configuration Setup
-----------------------------

Create a training configuration file. Here's a complete example:

.. code-block:: yaml

    # config.yaml
    base_llm: meta-llama/Llama-3.1-8B-Instruct
    base_model: null  # Set to checkpoint path to resume training
    attachment_token: <|reserved_special_token_0|>
    tokenizer_type: llama
    token_size: 4096
    
    # Data loaders for different modalities
    loaders:
      - loader_type: raw-image
        modality_type: image
    
    # Model modality configurations
    modalities:
      - model_type: meditron_clip
        clip_name: openai/clip-vit-large-patch14
        hidden_size: 4096
    
    # Training mode: ALIGNMENT, END2END, or FULL
    training_mode: ALIGNMENT
    
    # Dataset paths
    datasets:
      - packed_path: /path/to/your/training_dataset
    
    # Training arguments
    training_args:
      output_dir: ./checkpoints/multimeditron-trained
      dataloader_num_workers: 16
      dataloader_prefetch_factor: 4
      remove_unused_columns: false
      ddp_find_unused_parameters: false
      learning_rate: 1.0e-4
      bf16: true
      per_device_train_batch_size: 4
      gradient_accumulation_steps: 8
      num_train_epochs: 1
      gradient_checkpointing: true
      gradient_checkpointing_kwargs:
        use_reentrant: true
      save_strategy: epochs
      max_grad_norm: 1.0
      run_name: my-multimeditron-training
      deepspeed: ./deepspeed.json
      accelerator_config:
        dispatch_batches: false
      lr_scheduler_type: cosine_with_min_lr
      lr_scheduler_kwargs:
        min_lr: 3.0e-5
      report_to: wandb
      logging_steps: 1
      weight_decay: 0.01

DeepSpeed Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Create the DeepSpeed configuration file:

.. code-block:: json

    // deepspeed.json
    {
        "bf16": {
            "enabled": true
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": false,
            "contiguous_gradients": true,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "sub_group_size": 1e9,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": true
        },
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": false,
        "activation_checkpointing": {
            "partition_activations": false,
            "contiguous_memory_optimization": false,
            "cpu_checkpointing": false
        },
        "flops_profiler": {
            "enabled": false
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "single_submit": false,
            "overlap_events": false
        }
    }

Training Modes Explained
~~~~~~~~~~~~~~~~~~~~~~~~

- **ALIGNMENT**: Trains only the modality projection layers (fastest, recommended for starting)
- **END2END**: Trains modality encoders + projection layers + LLM (medium speed)
- **FULL**: End-to-end training with all parameters (slowest, best performance)

Step 4: Model Training
-----------------------

Single Node Training
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Set the number of GPUs
    export NPROC_PER_NODE=4
    
    # Launch training
    torchrun --nproc-per-node $NPROC_PER_NODE -m multimeditron train --config config.yaml

Multi-Node Training (SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a SLURM script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name multimeditron-training
    #SBATCH --output ~/reports/R-%x.%j.out
    #SBATCH --error ~/reports/R-%x.%j.err
    #SBATCH --nodes 2
    #SBATCH --ntasks-per-node 1
    #SBATCH --gres gpu:4
    #SBATCH --cpus-per-task 32
    #SBATCH --time 11:59:59
    #SBATCH --export=ALL
    
    echo "START TIME: $(date)"
    set -eo pipefail
    set -x
    
    GPUS_PER_NODE=4
    echo "NODES: $SLURM_NNODES"
    export HF_HOME=/path/to/hf/home
    
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=6200
    
    LAUNCHER="
      torchrun \
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $SLURM_NNODES \
      --node_rank \$SLURM_PROCID \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      --rdzv_backend c10d \
      --max_restarts 0 \
      --tee 3 \
      "
    
    export CMD="$LAUNCHER -m multimeditron train --config config.yaml"
    
    SRUN_ARGS=" \
      --cpus-per-task $SLURM_CPUS_PER_TASK \
      --jobid $SLURM_JOB_ID \
      --wait 60 \
      "
    
    srun $SRUN_ARGS bash -c "$CMD"
    echo "END TIME: $(date)"

Launch the training:

.. code-block:: bash

    sbatch training.sh

Monitoring Training
~~~~~~~~~~~~~~~~~~~

During training, you can monitor progress through:

1. **Console Output**: Real-time training metrics
2. **Weights & Biases**: If `report_to: wandb` is configured
3. **Checkpoint Files**: Saved in `output_dir` based on `save_strategy`

Resuming Training
~~~~~~~~~~~~~~~~~

To resume from a checkpoint, update your config:

.. code-block:: yaml

    base_model: ./checkpoints/multimeditron-trained/checkpoint-1000
    resume_from_checkpoint: true

Step 5: Inference
-----------------

Once training is complete, you can use your model for inference:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from multimeditron.model.model import MultiModalModelForCausalLM
    from multimeditron.model.data_loader import DataCollatorForMultimodal
    from multimeditron.dataset.loader import FileSystemImageLoader
    
    ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"
    
    # Load tokenizer and model
    model_path = "./checkpoints/multimeditron-trained"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = MultiModalModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.eval()
    
    # Prepare input
    modalities = [{"type": "image", "value": "path/to/test/image.jpg"}]
    conversations = [{
        "role": "user",
        "content": f"Describe this image: {ATTACHMENT_TOKEN}"
    }]
    
    sample = {
        "conversations": conversations,
        "modalities": modalities
    }
    
    # Create data collator
    loader = FileSystemImageLoader(base_path=".")
    collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        attachment_token=ATTACHMENT_TOKEN,
        chat_template=model.config.chat_template,
        modality_processors=model.processors(),
        modality_loaders={"image": loader},
        add_generation_prompt=True,
    )
    
    # Generate response
    batch = collator([sample])
    
    with torch.no_grad():
        outputs = model.generate(
            batch=batch, 
            temperature=0.7,
            max_new_tokens=512,
            do_sample=True
        )
    
    # Decode and print result
    result = tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )[0]
    
    print(result)

Troubleshooting Common Issues
------------------------------

Memory Issues
~~~~~~~~~~~~~

If you encounter GPU memory errors:

1. **Reduce batch size** in `training_args.per_device_train_batch_size`
2. **Increase gradient accumulation** to maintain effective batch size
3. **Enable gradient checkpointing** (already enabled in example)
4. **Use CPU offloading** in DeepSpeed config (already enabled)

Dataset Loading Errors
~~~~~~~~~~~~~~~~~~~~~~

Common dataset issues and solutions:

1. **Path not found**: Ensure dataset paths are absolute and accessible
2. **Format mismatch**: Verify your dataset matches the expected format
3. **Permission denied**: Check file permissions and Docker volume mounts

Training Stalls
~~~~~~~~~~~~~~~

If training appears stuck:

1. **Check logs**: Look for error messages in console output
2. **Verify network**: Ensure all nodes can communicate (multi-node)
3. **Monitor resources**: Use `nvidia-smi` to check GPU utilization

Performance Optimization Tips
------------------------------

1. **Use mixed precision** (bf16) for faster training
2. **Optimize data loading** with appropriate `dataloader_num_workers`
3. **Use DeepSpeed ZeRO-3** for large model training
4. **Enable gradient accumulation** for larger effective batch sizes
5. **Monitor with Weights & Biases** to track training metrics

Next Steps
----------

After completing this tutorial, you may want to:

1. **Fine-tune on specific domains** using your own datasets
2. **Experiment with different modalities** by adding new modality types
3. **Optimize hyperparameters** for better performance
4. **Deploy your model** using the provided inference scripts
5. **Contribute to MultiMeditron** by sharing your improvements

For more advanced topics, see:
- :ref:`configuration-label` for detailed configuration options
- :ref:`add-modality-label` for adding new modalities
- :ref:`dataset-format-label` for dataset format specifications