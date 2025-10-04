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

Raw modalities goes through 3 steps:

1. **Modality loading**: This steps loads modality from the dataset and transforms it into a raw modality format (for instance image bytes).
2. **Modality preprocessing**: This steps transforms raw modality into :code:`torch.Tensor`
3. **Modality embedding**: This steps is the :code:`forward` step of your modality embedder. It forwards the :code:`torch.Tensor` object of the preprocessing step to create a :code:`torch.Tensor`: the modality embedding.

Note that:

- Step 1 is **model agnostic**, every model uses the same loading functions.
- Step 2 and 3 are **model dependent**

This means that if you implement a model for an existing modality, you don't need to implement the modality loading step.

Implementation example
----------------------

To create a new modality embedder, you need to implement 3 classes:

- :class:`~multimeditron.dataset.loader.BaseModalityLoader` (only if implementing a new modality type): The modality loader to load the modality from the dataset
- :class:`~multimeditron.model.modalities.base.BaseModalityConfig`: The configuration file for both the processor and the modality model
- :class:`~multimeditron.model.modalities.base.BaseModalityProcessor`: The processor class to preprocess your modalities
- :class:`~multimeditron.model.modalities.base.BaseModality`: The modality model that forward your modalities


In this walkthrough, we will show how to load images and how to create a simple modality embedder.

Modality loader
^^^^^^^^^^^^^^^

Here is an example to load images from bytes:

.. code-block:: python

    from typing import Dict, Any, Union
    from multimeditron.dataset.loader import BaseModalityLoader, AutoModalityLoader
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

A modality loader should always inherit from `BaseModalityLoader` and be registered using the python annotation :meth:`multimeditron.model.modalities.AutoModalityLoader.register`

The `load` function has the following signature:
- Input: A dictionary that contains a key :code:`"value"`, i.e. `{"value" : <something>}`. This is the case for every modality. The actual format of the value field depends on the dataset format. See `TODO`
- Output returns the raw modality (here a `PIL.Image.Image`).


Modality configuration
^^^^^^^^^^^^^^^^^^^^^^

The configuration, processor, model architecture follows the same philosophy as `Huggingface custom model`_.

.. _Huggingface custom model: https://huggingface.co/docs/transformers/custom_models

The configuration file configures both the processor and the modality:

.. code-block:: python

    from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
    from multimeditron.model.modalities.base import BaseModality, BaseModalityConfig, AutoModality, BaseModalityProcessor
    from multimeditron.model.projectors.mlp import MLPProjector
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoConfig

    from typing import Dict, Any


    class ImageConfig(BaseModalityConfig):
        def __init__(
            self,
            hidden_size: int = 4096,
            max_batch_size: int = 32,
            clip_name: str = "openai/clip-vit-large-patch14",
            projection_type: str = "mlp",
            **kwargs
        ):
            super().__init__(
                max_batch_size=max_batch_size,
                modality_type="image",
                hidden_size=hidden_size,
                kwargs=kwargs
            )

            self.clip_name = clip_name
            self.projection_type = projection_type


