.. role:: bash(code)
   :language: bash

.. _deployment-label:

CSCS / SLURM deployment
=======================

This guide covers running MultiMeditron on the CSCS **Alps Clariden** cluster
(NVIDIA GH200, ARM64). It complements the generic :ref:`training <training-label>`
and :ref:`evaluation <evaluation-label>` guides with cluster-specific details.

Cluster conventions
--------------------

============================  ====================================================
Item                          Value
============================  ====================================================
Account                       ``a127``
Node type                     GH200, 4 × 96 GB GPU per node
Debug partition               max 2 nodes, 30 min wall time
Normal partition              up to 128 nodes
Training container (EDF)       ``~/.edf/multimeditron.toml``
Eval container (EDF)           ``~/.edf/multimeditron.toml`` (accelerate-based)
============================  ====================================================

Containers are launched via the EDF (Environment Definition File) mechanism with
:bash:`srun --environment=...`; the provided launchers handle this for you.

Environment setup
-----------------

The launchers export the variables training needs. The most important:

.. code-block:: bash

    export HF_HOME=/iopsstor/scratch/cscs/<user>/hf   # avoid home-dir quota errors
    export HF_TOKEN=<your_hf_token>                   # for gated model downloads
    export NCCL_NET_GDR_LEVEL=0                        # required on GH200 (see below)
    export PYTHONDONTWRITEBYTECODE=1                   # avoid stale .pyc on Lustre

Project-level secrets (HF/WandB tokens) live in a ``.env`` file at the repo root.

.. warning::

   Set ``HF_HOME`` to scratch. The home directory has a small quota and expert
   weight downloads will otherwise fail mid-run.

Launching training
-------------------

Use the generic launcher with the desired node count and config:

.. code-block:: bash

    # Stage 1 (alignment) — a few nodes is enough:
    sbatch --nodes 4 --time 06:00:00 sbatch_train.sh \
        cookbook/sft/moe/attn/pep/stage1_alignment.yaml

    # Stage 2 (end-to-end) — 128 nodes for the 7-expert ZeRO-3 run:
    sbatch --nodes 128 --time 12:00:00 sbatch_train.sh \
        cookbook/sft/moe/attn/pep/stage2_end2end.yaml

.. note::

   With ZeRO-3, the optimizer-state shard count is tied to the world size. A
   checkpoint sharded across N ranks must be **resumed at the same node count**.
   To resume, set ``resume_from_checkpoint: <path>`` in the YAML.

Launching evaluation
--------------------

Always use the accelerate-based :bash:`sbatch_eval.sh` (the vLLM path cannot load
the custom ``multimodal`` model type):

.. code-block:: bash

    export HF_TOKEN=<token>
    sbatch --time 03:00:00 --nodes 16 sbatch_eval.sh \
        <checkpoint_path> llama gmai,slake,path_vqa

Approximate wall time: ~50 min on 16 nodes, ~3.5 h on 4 nodes, for the three
standard benchmarks. Results land in
``reports/lmms_eval_results/<checkpoint_name>/``.

WandB offline sync
-----------------

Compute nodes have no outbound network. Runs log offline; sync them afterwards
from a debug job inside the container:

.. code-block:: bash

    wandb sync <run_dir>

Further reading:

- :any:`Training guide <training-label>`
- :any:`Evaluation guide <evaluation-label>`
- :any:`Troubleshooting <troubleshooting-label>`
