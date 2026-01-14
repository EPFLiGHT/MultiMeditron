.. _dataset-format-image:

Image modality
==============

Read :ref:`dataset-format-label` for more details about the general dataset format.

This section describes how to format image modalities in the dataset.

PIL Image format
----------------

Huggingface datasets automatically converts PIL images into bytes when saving to Arrow format. Therefore, you can directly use PIL images when creating your dataset. We provide an example below:

.. code-block:: python

    from PIL import Image
    import datasets

    def generate_sample(image_path):
        image = Image.open(image_path).convert("RGB")
        conversations = [
            {"role": "user", "content": "Describe the image: <|reserved_special_token_0|>."},
            {"role": "assistant", "content": "This is an image of ..."}
        ]
        return {
            "conversations": conversations,
            "modalities": [{"type": "image", "value": image}]
        }
        
    dataset = datasets.Dataset.from_generator(
        lambda: (generate_sample(path) for path in list_of_image_paths)
    )
    dataset.save_to_disk("path/to/save/dataset")

When saving the dataset to Arrow format, the images will be automatically converted to bytes with huggingface datasets.

When training the model, write in the configuration file

.. code-block:: yaml
    
    loaders:
        - loader_type: "raw-image"
          modality_type: "image"

    datasets:
        packed_path: "path/to/save/dataset"

Filesystem format
-----------------

You can also store the images on the filesystem and provide the paths to the images in the dataset. However, this format is not recommended for training as it is less efficient than storing the images directly in the dataset. We recommend using the PIL/Bytes format instead and to use the filesystem format only for inference.

In the dataset, the modality value must be the path to the image:

.. code-block:: python

    def generate_sample(image_path):
        conversations = [
            {"role": "user", "content": "Describe the image: <|reserved_special_token_0|>."},
            {"role": "assistant", "content": "This is an image of ..."}
        ]
        return {
            "conversations": conversations,
            "modalities": [{"type": "image", "value": image_path}]
        }

    dataset = datasets.Dataset.from_generator(
        lambda: (generate_sample(path) for path in list_of_image_paths)
    )
    dataset.save_to_disk("path/to/save/dataset")

When training the model, write in the configuration file

.. code-block:: yaml

    loaders:
        - loader_type: "fs-image"
          modality_type: "image"

    datasets:
        packed_path: "path/to/save/dataset"

    
