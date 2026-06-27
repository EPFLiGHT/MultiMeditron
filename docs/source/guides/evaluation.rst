.. _evaluation-label:

Evaluating a MultiMeditron model
=================================

This guide explains how to evaluate a trained MultiMeditron model using the
`lmms-eval <https://github.com/EPFLiGHT/lmms-eval>`_ framework. It covers
installation, running evaluations on single and multi-node setups, the available
benchmarks, and how to interpret results.


Overview
--------

MultiMeditron evaluation is built on top of the **lmms-eval** framework
(a fork maintained at `EPFLiGHT/lmms-eval <https://github.com/EPFLiGHT/lmms-eval>`_).
The framework registers a custom :code:`multimeditron` model type to load
MultiMeditron checkpoints and runs them against medical VQA benchmarks.

The supported benchmarks are:

* **GMAI** -- 4 550 samples with Chain-of-Thought (CoT) prompting
* **SLAKE** -- 642 samples covering radiology VQA (open-ended and yes/no)
* **PathVQA** -- approximately 6 700 samples for pathology VQA (open-ended and yes/no)

.. note::
   Additional benchmarks shipped with lmms-eval (e.g. VQAv2, MedQA, ScienceQA)
   are available but have not been validated with MultiMeditron.


Prerequisites
-------------

lmms-eval installation
~~~~~~~~~~~~~~~~~~~~~~

If you are using the MultiMeditron Docker image, lmms-eval is already installed
and you can skip this step.

Otherwise, install from the vendored copy or directly from GitHub:

.. code-block:: bash

   # Option A -- vendored third-party copy
   cd third-party/lmms-eval
   pip install -e .

   # Option B -- clone from GitHub
   git clone https://github.com/EPFLiGHT/lmms-eval.git
   cd lmms-eval
   pip install -e .

Make sure the MultiMeditron package itself is also installed (see :doc:`quickstart`).


Running evaluation
------------------

Single node (interactive)
~~~~~~~~~~~~~~~~~~~~~~~~~

Use :code:`accelerate launch` to run evaluation on a single node. Set
:code:`$NUM_PROC` to the number of GPUs available (check with :code:`nvidia-smi`).

.. code-block:: bash

   export PYTHONPATH=/path/to/MultiMeditron/src:/path/to/lmms-eval:${PYTHONPATH:-}

   python3 -m accelerate.commands.launch \
       --num_processes ${NUM_PROC:-4} \
       -m lmms_eval \
       --model multimeditron \
       --model_args pretrained=/path/to/checkpoint,tokenizer_type=llama,device_map=auto \
       --tasks gmai,slake,path_vqa \
       --batch_size 1 \
       --output_path ./eval_results

Replace the following placeholders:

* :code:`/path/to/checkpoint` -- path to the trained model checkpoint
* :code:`tokenizer_type` -- the tokenizer used during training (one of :code:`llama`, :code:`qwen3`, :code:`apertus`)
* :code:`--output_path` -- where the result JSON files will be written

You can find the correct :code:`tokenizer_type` by inspecting the training
configuration YAML:

.. code-block:: bash

   grep tokenizer_type config.yaml

Multi-node (SLURM)
~~~~~~~~~~~~~~~~~~~

For large-scale evaluation on a SLURM cluster, use the provided
:code:`sbatch_eval.sh` script. This script uses :code:`accelerate launch` under
the hood with :code:`srun` for multi-node orchestration.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   sbatch --time 03:00:00 --nodes 16 \
       sbatch_eval.sh /path/to/checkpoint llama gmai,slake,path_vqa

**Script arguments:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 50

   * - Argument
     - Default
     - Description
   * - :code:`checkpoint`
     - (required)
     - Absolute path to the model checkpoint directory
   * - :code:`tokenizer`
     - :code:`llama`
     - Tokenizer type: :code:`llama`, :code:`qwen3`, or :code:`apertus`
   * - :code:`tasks`
     - :code:`gmai,slake,path_vqa`
     - Comma-separated list of benchmark task names
   * - :code:`limit`
     - (none)
     - Limit number of samples per task (useful for quick tests)

**Quick debug run:**

.. code-block:: bash

   sbatch --partition=debug --nodes=2 --time=00:29:59 \
       sbatch_eval.sh /path/to/checkpoint llama gmai 20

This runs only 20 samples of GMAI on 2 nodes, useful for verifying the pipeline
works before committing to a full evaluation.

.. warning::
   The debug partition is limited to 2 nodes and 30 minutes wall time.


Node count and timing
~~~~~~~~~~~~~~~~~~~~~

The number of nodes directly affects evaluation time:

.. list-table::
   :header-rows: 1
   :widths: 20 30

   * - Nodes
     - Approximate wall time
   * - 4
     - ~3.5 hours
   * - 16
     - ~50 minutes


Available benchmarks
--------------------

GMAI
~~~~

General Medical AI benchmark with 4 550 samples. Uses Chain-of-Thought prompting
for complex medical reasoning tasks across multiple imaging modalities.

SLAKE
~~~~~

Semantics of Label, Knowledge, and Expertise -- a bilingual medical VQA dataset
with 642 evaluation samples. Contains both open-ended and yes/no questions
focused on radiology images.

PathVQA
~~~~~~~

Pathology Visual Question Answering with approximately 6 700 evaluation samples.
Covers histopathology images with both open-ended and yes/no question types.

.. note::
   Task definition YAML files are located in :code:`third-party/lmms-eval/lmms_eval/tasks/`
   under subdirectories :code:`gmai/`, :code:`slake/`, and :code:`path_vqa/`.


Reading results
---------------

Evaluation results are saved as JSON files under the output path. When using
:code:`sbatch_eval.sh`, results are written to:

.. code-block:: text

   ~/reports/lmms_eval_results/<checkpoint_name>/

Each run produces a timestamped JSON file containing per-task metrics (accuracy,
exact match, etc.) and per-sample predictions.

Logs from SLURM jobs are written to:

.. code-block:: text

   ~/reports/R-multimeditron-eval.<job_id>.out
   ~/reports/R-multimeditron-eval.<job_id>.err


Common issues
-------------

Do not use vLLM for evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   vLLM **cannot** load the custom :code:`multimodal` model type used by
   MultiMeditron. Always use the accelerate-based evaluation path
   (:code:`sbatch_eval.sh`). The vLLM-based evaluation script has been
   abandoned.

PYTHONPATH not set
~~~~~~~~~~~~~~~~~~

If you see :code:`ModuleNotFoundError: No module named 'multimeditron'` or
:code:`No module named 'lmms_eval'`, ensure both paths are on
:code:`PYTHONPATH`:

.. code-block:: bash

   export PYTHONPATH=/path/to/MultiMeditron/src:/path/to/lmms-eval:${PYTHONPATH:-}

HF_TOKEN not set
~~~~~~~~~~~~~~~~

The evaluation script requires :code:`HF_TOKEN` to be set in the environment.
If it is not set, the script will fail immediately with an error message. Export
it before submitting:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

Out of memory on large benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter OOM errors, increase the number of nodes. The evaluation
distributes samples across all available GPUs. Using 16 nodes (64 GPUs) is
recommended for the full benchmark suite.
