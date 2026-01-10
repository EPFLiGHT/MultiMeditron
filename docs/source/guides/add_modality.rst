.. role:: python(code)
   :language: python

.. _add-modality-label:

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

1. **Modality loading**: This step loads modality from the dataset and transforms it into a raw modality format (for instance image bytes).
2. **Modality preprocessing**: This step transforms raw modality into :code:`torch.Tensor`
3. **Modality embedding**: This step is the :code:`forward` step of your modality embedder. It forwards the :code:`torch.Tensor` object of the preprocessing step to create a :code:`torch.Tensor`: the modality embedding.

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
    from multimeditron.model.constants import MODALITY_VALUE_KEY
    import PIL
    import io

    @AutoModalityLoader.register("raw-image")
    class RawImageLoader(BaseModalityLoader):
        def __init__(self):
            super().__init__()

        def load(self, sample: Dict[str, Any]) -> PIL.Image.Image:
            image_bytes = sample[MODALITY_VALUE_KEY]["bytes"]
            image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image

A modality loader should always inherit from :class:`~multimeditron.dataset.loader.BaseModalityLoader` and be registered using the python annotation :meth:`~multimeditron.model.modalities.base.AutoModalityLoader.register`

The :code:`load` function has the following signature:

- Input: A dictionary that contains a key :code:`"value"`, i.e. :code:`{"value" : <something>}`. This is the case for every modality. The actual format of the value field depends on the dataset format. See `TODO`
- Output returns the raw modality (here a :class:`PIL.Image.Image`).


Modality configuration
^^^^^^^^^^^^^^^^^^^^^^

The configuration, processor, model architecture follows the same philosophy as `Huggingface custom model`_.

.. _Huggingface custom model: https://huggingface.co/docs/transformers/custom_models

The configuration file configures both the processor and the modality:

.. code-block:: python

    from multimeditron.model.modalities.base import BaseModality

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


Every configuration needs to inherit :class:`~multimeditron.model.modalities.base.BaseModalityConfig` and call the :code:`__init__` function from :code:`BaseModalityConfig` wth the arguments:

- :code:`max_batch_size`: the maximum amount of modalities that can be processed in a single batch by the `forward` function of the modality embedder
- :code:`modality_type`: which modality type does this processor/modality pair handle. This field should match the :code:`"type"` field in the dataset. See `TODO`
- :code:`hidden_size`: the projected shape of the modality embedder (i.e. the size of a LLM token embedding)

This configuration can be arbitrarily expanded with any JSON-serializable attributes. See `Huggingface custom model`_

Modality (pre)processor
^^^^^^^^^^^^^^^^^^^^^^^

A modality processor preprocess modalities to transform the raw modality from the loading step (here a :code:`PIL.Image.Image`) into a :code:`torch.Tensor`. This processing phase is applied during the collator phase (unlike the forward pass of the :class:`~multimeditron.model.modalities.base.BaseModality`)

.. code-block:: python

    from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
    from multimeditron.model.modalities.base import BaseModalityProcessor
    from transformers import AutoImageProcessor, AutoConfig

    from typing import Dict, Any

    class ImageProcessor(BaseModalityProcessor):
        def __init__(self, config):
            super().__init__(config)
            assert config.clip_name is not None, "clip_name must be specified in the config"

            self.image_processor = AutoImageProcessor.from_pretrained(config.clip_name)

            feature_extractor_config = AutoConfig.from_pretrained(config.clip_name, trust_remote_code=True)
            self._num_patches_per_entry = (feature_extractor_config.vision_config.image_size // feature_extractor_config.vision_config.patch_size) ** 2

        def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
            processed_modality = modality.copy()
            image = modality[MODALITY_VALUE_KEY]

            processed_modality[MODALITY_VALUE_KEY] = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
            processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

            return processed_modality


Each processor must inherit :class:`~multimeditron.model.modalities.base.BaseModalityProcessor` (which inherit from :class:`~transformers.ProcessorMixin`).

The modality processor must impement the :meth:`~multimeditron.model.modalities.base.BaseModalityProcessor.process` function. This function takes:

- A :code:`Dict`, this is exactly the output of the previous loading phase
- This function returns the exact same :code:`Dict` with the preprocessed modality in the :code:`"value"` key

Modality modeling
^^^^^^^^^^^^^^^^^

Lastly, we implement the modality model. This is the model that performs the forward pass during training. To optimize GPU throughput, you should only put operations that can be parallelized on GPU.

A modality class must inherit :class:`~multimeditron.model.modalities.base.BaseModality` is typically created with 2 main modules:

1. A pretrained modality embedder (like a CLIP model): This module produces meaningful embeddings for given modalities
2. A tunable projection module (usually a simple MLP or a linear layer): This module map embeddings from the modality embedder to the LLM embedding space. The dimension of this embedding space is given by the `hidden_size` attribute of :class:`~multimeditron.model.modalities.base.BaseModalityConfig`

.. code-block:: python

    from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
    from multimeditron.model.modalities.base import BaseModalityProcessor
    from transformers import AutoModel, AutoConfig
    import torch

    from typing import Dict, Any

    @AutoModality.register("meditron_clip")
    class ImageModality(BaseModality):
        config_class = ImageConfig
        preprocessor_class = ImageProcessor

        def __init__(self, config: ImageConfig):
            super().__init__(config)

            self.vision_tower_name = config.clip_name
            assert self.vision_tower_name is not None, "vision_tower_name must be specified in the config"

            self.feature_extractor = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
            self.embedding_size = self.feature_extractor.vision_embed_dim
            self._num_patches_per_entry = (self.feature_extractor.vision_model.config.image_size // self.feature_extractor.vision_model.config.patch_size) ** 2

            self.projector = MLPProjector(self.embedding_size, config.hidden_size, dtype=self.dtype)

        def forward(self, inputs: List[torch.Tensor]) -> torch.FloatTensor:
            inputs = torch.stack(inputs, dim=0)
            inputs = inputs.to(self.feature_extractor.device)
            image_features = self.feature_extractor.vision_model(inputs).last_hidden_state[:, 1:, :]

            projected = self.projector(image_features)

            return projected

        def freeze_modality_embedder(self):
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        def unfreeze_modality_embedder(self):
            for parameters in self.feature_extractor.parameters():
                parameters.requires_grad = True

        def unfreeze_projection(self):
            for parameters in self.projector.parameters():
                parameters.requires_grad = True


A modality class must implement 3 functions:

- :meth:`~multimeditron.model.modalities.base.BaseModality.forward`: this is the definition of the forward pass (which include the forward of both the modality embedder and the projection module)
- :meth:`~multimeditron.model.modalities.base.BaseModality.freeze_modality_embedder`: this function freezes the parameters of the modality embedder only
- :meth:`~multimeditron.model.modalities.base.BaseModality.unfreeze_modality_embedder`: this function unfreezes the parameters of the modality embedder
- :meth:`~multimeditron.model.modalities.base.BaseModality.unfreeze_projection`: this function unfreezes the parameters of the projection module 

Those "freezing" functions are used to train different part of the whole MultiMeditron architecture to ensure training stability.

TODO: Redirect to creating dataset + launching training
