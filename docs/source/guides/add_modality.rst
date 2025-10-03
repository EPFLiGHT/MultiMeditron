.. role:: python(code)
   :language: python

Adding new modality
===================

Structure of the repository:

.. code-block::

    src
    └── multimeditron
        ├── cli
        ├── config
        ├── dataset
        │   └── loader
        │       └── image
        ├── model
        │   ├── modalities
        │   └── projectors
        ├── train
        ├── utils
        └── verl

In order to add a new modality, we must first understand how the training pipeline process raw modalities:

.. graphviz::

   digraph G {
       rankdir=LR;

       A [label="Modality value in dataset"];
       B [label="Raw modality"];
       C [label="torch.Tensor"];
       D [label="torch.Tensor"];
        
       A -> B [label="modality loading"];
       B -> C [label="modality preprocessing"];
       C -> D [label="modality embedding"];
   }

Raw modalities goes through 2 steps:

1. **Modality loading**: This steps loads modality from the dataset and transforms it into a raw modality format (for instance image bytes).
2. **Modality preprocessing**: This steps transforms raw modality into `torch.Tensor`
3. **Modality embedding**: This steps is the `forward` step of your modality embedder. It forwards the `torch.Tensor` object of the preprocessing step to create a `torch.Tensor`: the modality embedding.

Note that:
- Step 1 is **model agnostic**, every model uses the same loading functions.
- Step 2 and 3 are **model dependent**

This means that if you implement a model for an existing modality, you don't need to implement the modality loading step.

Implementation example
----------------------

To create a new modality embedder, you need to implement 3 classes:

- :class:`multimeditron.dataset.loader.BaseModalityLoader` (only if implementing a new modality type): The modality loader to load the modality from the dataset
- :class:`multimeditron.model.modalities.base.BaseModalityConfig`: The configuration file for both the processor and the modality model
- :class:`multimeditron.model.modalities.base.BaseModalityProcessor`: The processor class to preprocess your modalities
- :class:`multimeditron.model.modalities.base.BaseModality`: The modality model that forward your modalities


In this walkthrough, we will show how to load images and how to create a simple modality embedder.

Modality loader
^^^^^^^^^^^^^^^

Here is an example to load images from bytes:

.. code-block:: python

    import os
    from typing import Dict, Any, Union
    from multimeditron.dataset.loader import BaseModalityLoader, AutoModalityLoader
    import pathlib
    import numpy as np
    import PIL
    import io

    @AutoModalityLoader.register("raw-image")
    class RawImageLoader(BaseModalityLoader):
        def __init__(self):
            super().__init__()

        def load(self, sample: Dict[str, Any]) -> PIL.Image.Image:
            image_bytes = sample["value"]["bytes"]
            image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image


The `load` function takes a dictionary that contains a key `"value"` and returns the raw modality (here a `PIL.Image.Image`).

Modality configuration
^^^^^^^^^^^^^^^^^^^^^^

The configuration, processor, model architecture follows the same philosophy as `Huggingface custom model`_.

.. _Huggingface custom model: https://huggingface.co/docs/transformers/custom_models

The configuration file configures both the processor and the modality:



