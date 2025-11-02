Dataset format
==============

This section describes the dataset format supported by the MultiMeditron pipeline. The dataset format varies from one modality to another and you can add your own modality by following :any:`this tutorial <add-modality-label>`.

Our training pipeline supports two types of dataset: pretraining and instruction-tuning datasets. Here are the two formats that we support:

1. Arrow/Parquet format (recommended): where the modalities are directly stored in the dataset
2. JSONL format (not recommended): where the images and modalities are stored on the file system. Those dataset must be processed with :code:`merge_inputs.py`

Arrow format (recommended) 
--------------------------

Pretraining dataset
^^^^^^^^^^^^^^^^^^^

Each dataset must contain a column :code:`text` and a column :code:`modalities`. The :code:`text` column contains string of the following form:

.. code-block:: python

    "Let's compare the first image: <|reserved_special_token_0|>, and the second 3D image: <|reserved_special_token_0|>"

And the code:`modalities` column must be of the following form:

.. code-block:: python

    [{"type": "modality_type", "value" : some_modality}]

For instance, for image type, :code:`some_modality` must contains the bytes of the image

Note that we use a special placeholder :code:`<|reserved_special_token_0|>` to indicate the position of the tokens from the modality

Instruction-tuning dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

It's the same as the pretraining dataset but instead of the :code:`text` column, we have a :code:`conversations` column:

.. code-block:: python

    [
        {"role" : "system", "content" : "You are Meditron"},
        {"role" : "user", "content" : "Compare the CT scan <|reserved_special_token_0|> with the image <|reserved_special_token_0|>."},
        {"role" : "assistant", "content" : "Lorem ipsum dolor sit amet, consectetur adipiscing elit."},
        {"role" : "user", "content" : "How is it related to that signal: <|reserved_special_token_0|>?"},
        {"role" : "assistant", "content" : "Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor."}
    ]

JSONL format
------------

We also support :code:`.jsonl` files where each line corresponds to a sample. We describe how each sample must be formatted:

.. warning::

   Please note that JSONL format is not recommended! We provide scripts to convert JSONL-formatted dataset into Arrow dataset. If your dataset is  in a JSONL format, you need to convert it first to Arrow before training.


Pretraining format
^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "text": "Let's compare the first image: <|reserved_special_token_0|>, and the second 3D image: <|reserved_special_token_0|>",
      "modalities": [{"type" : "image", "value" : "path/to/png"}, {"type" : "image_3d", "value" : "path/to/npy"}]
    }

Instruction-tuning format
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
      "conversations": [
        {"role": "system", "content" : "You are Meditron"},
        {"role": "user", "content" : "Compare the CT scan <|reserved_special_token_0|> with the image <|reserved_special_token_0|>."},
        {"role": "assistant", "content" : "Lorem ipsum dolor sit amet, consectetur adipiscing elit."},
        {"role": "user", "content" : "How is it related to that signal: <|reserved_special_token_0|>?"},
        {"role": "assistant", "content" : "Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor."}
      ],
      "modalities": [{"type" : "image_3d", "value" : "path/to/npy"}, {"type" : "image", "value" : "path/to/png"}, {"type" : "signal", "value" : "path/to/npy"}]
    }

