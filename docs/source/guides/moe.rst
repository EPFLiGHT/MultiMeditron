.. _moe-label:

Mixture-of-Experts (MoE) Architecture
======================================

MultiMeditron uses a **Mixture-of-Experts (MoE)** vision encoder that routes
each input image through multiple domain-specific CLIP models and fuses their
outputs before projecting into the language model's token space. This page
explains the architecture, supported fusion and projection strategies, gating
mechanism, configuration, and training stages.

Overview
--------

Instead of relying on a single vision encoder, MultiMeditron maintains a pool
of specialist CLIP encoders, each fine-tuned on a different medical imaging
domain:

* **CT** -- Computed tomography scans
* **MRI** -- Magnetic resonance images
* **Ultrasound** -- Sonographic images
* **X-ray** -- Chest and skeletal radiographs
* **General** -- Natural and mixed-domain images (base CLIP)
* **Ophthalmology** -- Fundus photographs and OCT scans
* **Dermatology** -- Skin lesion images

A lightweight **gating network** scores each expert for a given image. The
top-k experts (by gating score) are activated, their patch embeddings are fused
according to a configurable strategy, and the result is projected into the
LLM embedding space.


Architecture
------------

The data flow through the MoE vision module is as follows:

.. code-block:: text

   Input image (224x224)
         |
   Gating Network (ResNet-50)
         |
   +-----+------------------------------------------+
   | Expert CLIP 1 (CT)                              |
   | Expert CLIP 2 (MRI)                             |
   | Expert CLIP 3 (Ultrasound)                      |
   | Expert CLIP 4 (X-ray)                           |
   | Expert CLIP 5 (General)                          |
   | Expert CLIP 6 (Ophthalmology)                    |
   | Expert CLIP 7 (Dermatology)                      |
   +-----+------------------------------------------+
         |  top-k selected, softmax-weighted
         |
   Fusion (cross-attention or weighted average)
         |
   Projection (per-expert or shared MLP)
         |
   LLM token embeddings --> LLaMA 3.1 8B (Meditron3)

**Step by step:**

1. The **gating network** processes the raw image and produces a softmax weight
   vector of shape ``(batch, num_experts)``.
2. All expert CLIP encoders run on the image in parallel.  Each expert outputs
   patch embeddings of shape ``(batch, num_patches, expert_dim)``, with the CLS
   token stripped.
3. A **fusion module** combines the expert outputs into a single sequence of
   patch embeddings using the gating weights.
4. A **projection layer** maps the fused (or per-expert) embeddings to the
   hidden size expected by the language model.
5. The projected embeddings replace the image placeholder tokens in the LLM
   input sequence, and normal causal language modelling proceeds.


Fusion Methods
--------------

The fusion strategy is controlled by the ``fusion_method`` field in the
modality configuration.

Cross-Attention (``cross_attn``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **generalist** expert's patch tokens serve as queries, while the specialist
experts' tokens (weighted by gating scores) serve as key/value context.  A
multi-head cross-attention layer produces a fused representation of shape
``(batch, num_patches, hidden_size)``.

This is the recommended method and is used by all ``ATTN-*`` experiments listed
in the cookbook.

.. note::

   The generalist expert is identified by ``generalist_idx`` in the config.
   Setting it to ``-1`` uses the last entry in ``expert_clip_names``.


Weighted Average (``weighted_average``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each expert's patch embeddings are scaled by the corresponding gating weight
and summed element-wise.  The result has the same shape as a single expert's
output.  This is computationally cheaper but less expressive than
cross-attention.


Sequence Append (``sequence_append``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All top-k experts' patch embeddings are concatenated along the sequence
dimension, producing a token sequence that is ``top_k * num_patches`` long.
The LLM sees more tokens per image, which increases memory usage proportionally.


Projection Strategies
---------------------

Per-Expert Projection (PEP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each expert has its **own** MLP projector that maps from the expert's native
embedding dimension to the LLM hidden size.  Projection happens *before*
fusion, so cross-attention operates in the LLM embedding space.

Use the model type ``moe_meditron_clip_pep`` in the YAML config:

.. code-block:: yaml

   modalities:
     - model_type: moe_meditron_clip_pep
       hidden_size: 4096
       # ...

PEP allows each expert to learn its own mapping independently, which is
especially useful when experts have different pre-training distributions (e.g.
pathology vs. radiology).


Shared Projection
^^^^^^^^^^^^^^^^^

A single MLP projector is shared across all experts.  Fusion happens in the
expert's native embedding space and the shared projector maps the fused output
to the LLM hidden size.

Use the model type ``moe_meditron_clip`` in the YAML config:

.. code-block:: yaml

   modalities:
     - model_type: moe_meditron_clip
       hidden_size: 4096
       # ...

.. warning::

   Shared projection requires all experts to have the **same** native embedding
   dimension.  If you mix CLIP architectures with different hidden sizes, use
   PEP instead.


Gating Mechanism
----------------

The gating network is a **ResNet-50** backbone with a replaced
fully-connected head that produces ``num_classes`` logits (one per expert).
Softmax is applied to obtain per-expert weights, and ``torch.topk`` selects
the ``top_k`` experts to activate.

.. code-block:: text

   Input image (224x224)
         |
   ResNet-50 (frozen or fine-tuned)
         |
   Linear(2048, num_experts)
         |
   Softmax --> per-expert weights
   Top-K   --> selected expert indices

The gating network is trained as a **standalone image classifier** before any
MoE training begins, using an ImageFolder dataset with one subdirectory per
expert class.  After training, it is converted to HuggingFace format
(``GatingNetwork`` / ``GatingNetworkConfig``) so it can be loaded via
``from_pretrained()`` and referenced by the ``gating_path`` config field.

.. note::

   Gating routing accuracy directly impacts MoE quality.  If the gating
   network achieves less than ~80% classification accuracy, experts receive
   off-modality images and the model can underperform a single-encoder
   baseline.

See ``cookbook/gating/README.md`` for the step-by-step gating network training,
conversion, and debugging procedure.


Configuration
-------------

Below is an annotated example of a MoE modality block using cross-attention
fusion with per-expert projection.  This block goes inside the ``modalities``
list of a training YAML config (see :ref:`config-ref-label` for the full
schema).

.. code-block:: yaml

   modalities:
     - model_type: moe_meditron_clip_pep        # PEP variant (or moe_meditron_clip for shared)
       image_processor: /path/to/clip-vit-base-patch32
       hidden_size: 4096                         # Must match the LLM hidden size

       expert_clip_names:                        # One entry per expert CLIP model
         - /path/to/MedExpert-CT
         - /path/to/MedExpert-MRI
         - /path/to/MedExpert-Ultrasound
         - /path/to/MedExpert-Xray
         - /path/to/clip-vit-base-patch32        # General (base CLIP)
         - /path/to/OphthalmologyExpert
         - /path/to/SkinExpert

       generalist_idx: 4                         # Index of the general-purpose expert
       gating_path: /path/to/MultiMeditron-Gating  # Pretrained gating network
       fusion_method: cross_attn                 # cross_attn | weighted_average | sequence_append
       top_k_experts: 3                          # Number of experts to activate per image

**Key fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``model_type``
     - --
     - ``moe_meditron_clip_pep`` (per-expert projection) or ``moe_meditron_clip`` (shared projection).
   * - ``expert_clip_names``
     - ``[]``
     - Ordered list of paths or HuggingFace model IDs for each expert CLIP model.
   * - ``gating_path``
     - ``""``
     - Path to a pretrained ``GatingNetwork``.  If empty, uniform weights are used.
   * - ``fusion_method``
     - ``weighted_average``
     - One of ``cross_attn``, ``weighted_average``, or ``sequence_append``.
   * - ``top_k_experts``
     - ``5``
     - Number of top experts to activate.  Lower values sharpen routing.
   * - ``generalist_idx``
     - ``-1``
     - Index into ``expert_clip_names`` pointing to the general-purpose CLIP.  ``-1`` means the last entry.
   * - ``hidden_size``
     - ``4096``
     - Output embedding dimension; must match the base LLM hidden size.
   * - ``cross_attn_heads``
     - ``8``
     - Number of attention heads in the cross-attention fusion layer (only used when ``fusion_method`` is ``cross_attn``).


Training Stages
---------------

MoE training follows a three-phase pipeline.  The gating network is trained
first as a prerequisite, then the full model is trained in two stages.

Phase 0: Gating Network Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train a ResNet-50 classifier to route images to the correct expert.  This is a
lightweight step (single GPU, ~30 min) and produces the ``gating_path``
checkpoint consumed by the MoE configs.

See ``cookbook/gating/README.md`` for the full procedure.


Phase 1: Alignment (Stage 1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alignment trains the projection layers and cross-attention module while keeping
the expert CLIP encoders and the LLM backbone **frozen**.  The goal is to teach
the model to interpret expert embeddings without forgetting language
capabilities.

Set ``training_mode: ALIGNMENT`` in the YAML config.

.. code-block:: yaml

   training_mode: ALIGNMENT

   modalities:
     - model_type: moe_meditron_clip_pep
       top_k_experts: 5            # Higher top-k for broader exposure during alignment
       # ... (expert list, gating_path, fusion_method as above)

   training_args:
     num_train_epochs: 3
     learning_rate: 1.0e-5
     per_device_train_batch_size: 8
     gradient_accumulation_steps: 4

**What is frozen:**

* All expert CLIP encoder parameters
* All LLM parameters
* Gating network parameters

**What is trained:**

* Per-expert MLP projectors (or shared projector)
* Cross-attention layer (if ``fusion_method: cross_attn``)

Typical training scale: 8 nodes, ~4--6 hours for 3 epochs.

For the full Stage 1 config, see ``cookbook/sft/moe/attn/pep/stage1_alignment.yaml``.


Phase 2: End-to-End Fine-Tuning (Stage 2)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

End-to-end training unfreezes the entire model (LLM + projectors +
cross-attention) and trains on richer instruction-tuning and medical VQA data.
This is the most compute-intensive phase.

Set ``training_mode: END2END`` and point ``base_model`` to the Stage 1
checkpoint:

.. code-block:: yaml

   base_model: /path/to/stage1/checkpoint
   training_mode: END2END

   modalities:
     - model_type: moe_meditron_clip_pep
       top_k_experts: 3            # Lower top-k for sharper routing during fine-tuning
       # ...

   training_args:
     num_train_epochs: 1
     learning_rate: 1.0e-5
     per_device_train_batch_size: 8
     gradient_accumulation_steps: 2
     save_steps: 50                # Save frequently -- ZeRO-3 checkpoints are large

.. warning::

   Stage 2 with ZeRO-3 ties checkpoint shards to the exact GPU rank count.
   Always resume with the **same** number of nodes and GPUs as the original
   run.  If you need to change scale, convert the checkpoint first using
   DeepSpeed's ``zero_to_fp32.py``.

Typical training scale: 128 nodes, ~23 hours for 1 epoch.

For the full Stage 2 config, see ``cookbook/sft/moe/attn/pep/stage2_end2end.yaml``.


Experiment Results
------------------

The following table summarises published results for the main MoE configurations
alongside single-encoder baselines.  All models use LLaMA 3.1 8B as the base
LLM.

.. list-table::
   :header-rows: 1
   :widths: 35 8 10 10 10 10 10

   * - Model
     - GMAI
     - PathVQA y/n
     - PathVQA open
     - PathVQA all
     - SLAKE y/n
     - SLAKE all
   * - CLIP (single encoder)
     - 34.0
     - 60.6
     - 5.6
     - 33.1
     - 50.5
     - 30.3
   * - ATTN-PEP (MoE)
     - 29.6
     - 59.1
     - 1.5
     - 30.3
     - 51.1
     - 29.6
   * - ATTN-SHARED (MoE)
     - 28.6
     - 56.9
     - 2.0
     - 29.5
     - 46.0
     - 27.5
   * - AVG-PEP (MoE)
     - 30.7
     - 46.5
     - 2.5
     - 24.5
     - 47.6
     - 27.6
   * - AVG-SHARED (MoE)
     - 29.7
     - 46.8
     - 2.6
     - 24.2
     - 49.5
     - 25.8

See the :ref:`training guide <training-label>` for general training instructions
and ``cookbook/README.md`` for the full experiment matrix.


Further Reading
---------------

* :ref:`training-label` -- General training setup and launch instructions.
* :ref:`config-ref-label` -- Full configuration reference.
* ``cookbook/gating/README.md`` -- Step-by-step gating network training guide.
* ``cookbook/sft/moe/`` -- MoE training YAML configs for all fusion/projection
  combinations.
