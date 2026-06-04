.. role:: bash(code)
   :language: bash

.. role:: yaml(code)
   :language: yaml

.. _add-expert-label:

Adding a new expert
===================

This guide explains how to add a new **CLIP expert encoder** to the
Mixture-of-Experts (MoE) model. This is different from
:ref:`adding a new modality <add-modality-label>`: a new *modality* introduces a
new input type (e.g. audio), whereas a new *expert* adds another specialist
encoder for the **existing** image modality (e.g. a dedicated Dermatology or
Ophthalmology encoder).

The worked example below is the extension from 5 to 7 experts (adding the
Ophthalmology and Dermatology experts). See :ref:`moe-label` for the MoE
architecture itself.

Overview
--------

Adding an expert touches four things, in order:

1. Obtain the expert CLIP encoder (train or download it).
2. Add it to the ``expert_clip_names`` list in the MoE training config.
3. Retrain the gating network so it has a class for the new expert.
4. Re-run Stage 1 (alignment) and Stage 2 (end-to-end) training.

Because the number of experts changes, the per-expert projectors and the gating
output dimension change too — so existing checkpoints are **not** shape-compatible
and the model must be retrained from the alignment stage.

1. Obtain the expert encoder
----------------------------

Train a CLIP encoder specialised on the target domain, or use an existing one.
The canonical expert trainer is invoked through the CLI:

.. code-block:: bash

    multimeditron train-expert scripts/config_us.yaml

The config selects the vision/text backbones and the dataset mixture. See
``scripts/README.md`` for details. Place the resulting checkpoint somewhere
referenceable from the cluster (e.g. ``models/CLIP/<MyExpert>``).

2. Register the expert in the MoE config
----------------------------------------

Add the expert checkpoint path to ``expert_clip_names`` in both the Stage 1 and
Stage 2 configs (``cookbook/sft/moe/attn/pep/stage1_alignment.yaml`` and
``stage2_end2end.yaml``):

.. code-block:: yaml

    modalities:
      - model_type: moe_meditron_clip_pep
        image_processor: .../clip-vit-base-patch32
        hidden_size: 4096
        expert_clip_names:
          - .../MedExpert-CT
          - .../MedExpert-MRI
          - .../UltraSoundCLIP
          - .../MedExpert-Xray
          - .../clip-vit-base-patch32        # Generalist
          - .../OphthalmologyExpert
          - .../SkinExpert                   # <-- newly added experts
        generalist_idx: -1                   # index of the generalist (-1 = last)
        gating_path: .../MultiMeditron-Gating
        fusion_method: cross_attn            # cross_attn | avg | cat
        top_k_experts: 3

.. note::

   The **order** of ``expert_clip_names`` defines the expert index. It must match
   the class order used when training the gating network (next step), otherwise
   images will be routed to the wrong encoder.

3. Retrain the gating network
------------------------------

The gating network is a ResNet50 classifier whose output dimension equals the
number of experts. Adding an expert means adding a class, so it must be
retrained. Add the new class to ``dataset_class_map`` in
``config/gating_7class.yaml`` (mapping the new class index to one or more Arrow
datasets representative of that domain), keeping ``class_names`` aligned with
``expert_clip_names`` above, then:

.. code-block:: bash

    torchrun --nproc_per_node=4 scripts/train_gating.py --config config/gating_7class.yaml
    # or, on CSCS:  sbatch sbatch_train_gating.sh

The script saves a ready-to-use HuggingFace ``GatingNetwork`` at ``output_dir``;
point ``gating_path`` at it. See ``cookbook/gating/README.md`` for the full guide.

4. Re-run alignment and end-to-end training
--------------------------------------------

Run Stage 1 (projectors only) then Stage 2 (end-to-end) — see
:ref:`training-label`:

.. code-block:: bash

    sbatch --nodes 4 sbatch_train.sh cookbook/sft/moe/attn/pep/stage1_alignment.yaml
    sbatch --nodes 128 sbatch_train.sh cookbook/sft/moe/attn/pep/stage2_end2end.yaml

5. Verify routing
-----------------

Confirm each modality routes to its dedicated expert with the routing analysis
script (no GPU required):

.. code-block:: bash

    python3 scripts/gating_routing_analysis.py
    # or:  sbatch sbatch_gating_analysis.sh

Pitfalls
--------

The 5 → 7 expert extension surfaced two bugs worth watching for when the expert
count changes:

- **fp32 fallback**: pass ``dtype=`` (not ``torch_dtype=``) to
  ``AutoConfig.from_pretrained`` — the wrong keyword is silently ignored and the
  model loads in fp32.
- **CUDA index-out-of-bounds**: changing the expert count can desynchronise the
  gating output size and the expert list; ensure the gating ``num_classes`` and
  ``len(expert_clip_names)`` match.

Further reading:

- :any:`Mixture-of-Experts architecture <moe-label>`
- :any:`Launching a training <training-label>`
- :any:`CSCS deployment <deployment-label>`
