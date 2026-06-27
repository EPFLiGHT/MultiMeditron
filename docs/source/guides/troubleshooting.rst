.. role:: bash(code)
   :language: bash

.. _troubleshooting-label:

Troubleshooting
===============

Common failure modes when training and evaluating MultiMeditron on the CSCS
GH200 cluster, with their fixes. See also :ref:`known-issues <docker-permission>`
for installation issues and :ref:`deployment <deployment-label>` for the launch
environment.

Training hangs on the first NCCL collective
--------------------------------------------

**Symptom**: multi-node training stalls indefinitely at the first allreduce /
allgather, with no error.

**Cause**: GPUDirect RDMA over the Slingshot interconnect fails on Clariden GH200
nodes.

**Fix**: set ``NCCL_NET_GDR_LEVEL=0``. The launchers export this already; if you
write your own, add it to the environment:

.. code-block:: bash

    export NCCL_NET_GDR_LEVEL=0

A code patch in ``trainer.py`` is silently not applied
------------------------------------------------------

**Symptom**: a change to a Python module appears to have no effect on compute
nodes (e.g. the sequence-packing FA2 patch does not activate).

**Cause**: the Lustre filesystem can serve stale ``.pyc`` bytecode to compute
nodes.

**Fix**: disable bytecode caching:

.. code-block:: bash

    export PYTHONDONTWRITEBYTECODE=1

Resume fails with a shard-count mismatch
----------------------------------------

**Symptom**: resuming a ZeRO-3 run from a checkpoint fails or loads incorrectly.

**Cause**: the optimizer-state shards are partitioned across the world size, so a
checkpoint saved on N ranks expects N ranks on resume.

**Fix**: resume at the **same node count** used to create the checkpoint (e.g.
128 nodes for the 7-expert Stage 2 run).

Evaluation cannot load the model
--------------------------------

**Symptom**: vLLM-based eval errors out loading the checkpoint.

**Cause**: vLLM cannot load the custom ``multimodal`` model type.

**Fix**: use the accelerate-based :bash:`sbatch_eval.sh`. The vLLM eval path has
been abandoned.

``decord`` import error during evaluation
-----------------------------------------

**Symptom**: eval crashes with ``ModuleNotFoundError: No module named 'decord'``.

**Cause**: the eval container does not ship ``decord`` (and it is unavailable on
ARM64).

**Fix**: it is lazy-imported inside ``try/except`` where actually needed — do not
add ``decord`` to install lists or import it at module top-level.

HuggingFace cache quota errors
------------------------------

**Symptom**: downloads fail partway with a disk-quota error.

**Cause**: ``HF_HOME`` points at the small-quota home directory.

**Fix**: point it at scratch: ``export HF_HOME=/iopsstor/scratch/cscs/<user>/hf``.

Model silently loads in fp32
----------------------------

**Symptom**: unexpected memory use / mixed-precision behaviour.

**Cause**: ``AutoConfig.from_pretrained(..., torch_dtype=...)`` — the wrong
keyword is silently ignored.

**Fix**: pass ``dtype=`` instead of ``torch_dtype=``.

``envsubst`` breaks SLURM variables
-----------------------------------

See :ref:`known-issues <docker-permission>`. ``envsubst`` blanks out
``$SLURM_*`` variables; pass values via :bash:`sbatch` CLI overrides instead of
piping launcher scripts through it.
