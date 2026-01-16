.. _training-label:

Training a MultiMeditron model 
==============================


This tutorial provides a step-by-step guide on how to train a model using MultiMeditron. We will walk you through the process with clear examples.

Configuration files
-------------------

Each training is configured through a YAML file. To get the full documentation of the different arguments supported by the configuration file, refer to :any:`the configuration reference <config-ref-label>`

.. code-block:: yaml

    base_llm: Qwen/Qwen3-8B
    base_model: null
    attachment_token: <|reserved_special_token_0|>
    tokenizer_type: qwen3
    token_size: 4096
    
    loaders:
      - loader_type: raw-image
        modality_type: image
    
    modalities:
      - model_type: meditron_clip
        clip_name: openai/clip-vit-large-patch14
        hidden_size: 4096
    
    training_mode: ALIGNMENT
    
    datasets:
      - packed_path: /path/to/dataset
    
    training_args:
      output_dir: /path/to/checkpoint
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
      deepspeed: deepspeed.json
      accelerator_config:
        dispatch_batches: false
      lr_scheduler_type: cosine_with_min_lr
      lr_scheduler_kwargs:
        min_lr: 3.0e-5
      logging_steps: 1
      weight_decay: 0.01
 

Make sure to replace :code:`/path/to/dataset` and :code:`/path/to/checkpoint` by your dataset and the actual output checkpoint path. 
Store this file in a YAML file. In our case, we store it in :code:`config.yaml`.

Additionally, we are using Deepspeed for parallelism and we need to create a deepspeed config. Here is our config used on a NVidia GH200 setup:

.. code-block:: json

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

Store this file in :code:`deepspeed.json` file, make sure that the path to this file matches the :code:`training_args.deepspeed` argument from the YAML configuration.


Launch the training
-------------------

Once the training configuration are done, we are ready to launch a training. We support both single node and multi node training.

Single node training
^^^^^^^^^^^^^^^^^^^^

To launch a single training, run the following command

.. code-block:: bash

    torchrun --nproc-per-node $PROC_PER_NODE -m multimeditron train --config config.yaml

where :code:`$PROC_PER_NODE` is the number of GPUS available

Multi node training
^^^^^^^^^^^^^^^^^^^

We provide scripts to launch MultiMeditron training on multi node cluster. We provide scripts to launch trainings on:

* SLURM cluster
* TODO: Provide script for Run:ai cluster

SLURM cluster
"""""""""""""

To launch a training on a SLURM cluster, we can use the following :code:`sbatch` script:

.. code-block:: bash

   #!/bin/bash
    #SBATCH --job-name multimeditron-training
    #SBATCH --output ~/reports/R-%x.%j.out
    #SBATCH --error ~/reports/R-%x.%j.err
    #SBATCH --nodes 4         # number of Nodes
    #SBATCH --ntasks-per-node 1     # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
    #SBATCH --gres gpu:4        # Number of GPUs
    #SBATCH --cpus-per-task 288     # number of CPUs per task.
    #SBATCH --time 11:59:59       # maximum execution time (DD-HH:MM:SS)
    #SBATCH --export=ALL,SCRATCH=/iopsstor/scratch/cscs/$USER
    #SBATCH -A a127

    echo "START TIME: $(date)"
    # auto-fail on any errors in this script
    set -eo pipefail
    # logging script's variables/commands for future debug needs
    set -x
    ######################
    ### Set enviroment ###
    ######################
    GPUS_PER_NODE=4
    echo "NODES: $SLURM_NNODES"
    export HF_HOME=/path/to/hf/home
    ######################
    #### Set network #####
    ######################
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    MASTER_PORT=6200
    ######################
    # note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
    # 0 and the launcher will hang
    #
    # same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time

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

    echo $CMD

    SRUN_ARGS=" \
      --cpus-per-task $SLURM_CPUS_PER_TASK \
      --jobid $SLURM_JOB_ID \
      --wait 60 \
      -A a127 \
      --reservation=sai-a127
      --environment ~/.edf/multimodal.toml
      "

    # bash -c is needed for the delayed interpolation of env vars to work
    srun $SRUN_ARGS bash -c "$CMD"
    echo "END TIME: $(date)"

..

Make sure to set your :code:`$HF_HOME` properly before launching the training. Models and datasets will be downloaded in this folder which can take many GB! Save this configuration file in a file called :code:`training.sh`

Finally launch the training by running this command: 

.. code-block:: bash

   sbatch training.sh
