.. role:: bash(code)
   :language: bash

Quickstart
==========

This quickstart guide will help you get started with MultiMeditron.

Installation
------------

To install MultiMeditron, you can either use pip or a prebuilt docker image.

Pip Installation
~~~~~~~~~~~~~~~~
You can install MultiMeditron using pip. Note that this installation supposes that you already have :code:`torch` installed! 
If not, go on the following `link`_, select the relevant torch version and your CUDA driver version and run the given command.

.. _link: https://pytorch.org/get-started/locally/

Then run the following command in your terminal:

.. code-block:: bash

    git clone https://github.com/EPFLiGHT/MultiMeditron.git
    cd MultiMeditron 
    pip install -e ".[flash-attn]"


Docker Installation
~~~~~~~~~~~~~~~~~~~

You can pull the latest MultiMeditron docker image from Docker Hub. Run the following command in your terminal:

.. tabs::

    .. tab:: AMD64

        .. code-block:: bash

            docker pull michelducartier24/multimeditron-git:latest-amd64
    
    .. tab:: ARM64

        .. code-block:: bash
            
            docker pull michelducartier24/multimeditron-git:latest-arm64


We also provide Docker images that runs on specific versions of the GitHub repository:

.. tabs::

    .. tab:: AMD64

        .. code-block:: bash

            docker pull michelducartier24/multimeditron-git:<commit-hash>-amd64
    
    .. tab:: ARM64

        .. code-block:: bash
            
            docker pull michelducartier24/multimeditron-git:<commit-hash>-arm64

Retrieve the commit hash that you need, and replace the `<commit-hash>` placeholder with the real commit hash.

.. note::
   On certain systems with custom permissions on volumes, the Docker image won't give you enough permissions or won't give you a username for the UID because they are not included in :bash:`/etc/passwd`.
   Please check :ref:`docker-permission`

MultiMeditron inference
-----------------------

Once you have installed MultiMeditron, you can run inference on your images. Here is an example script to run inference using the pip installation. In this example, we are loading a model based on Llama 3.1 model.

.. code-block:: python

    import torch
    from transformers import AutoTokenizer 
    import logging
    import os

    from multimeditron.dataset.loader import FileSystemImageLoader
    from multimeditron.model.model import MultiModalModelForCausalLM 
    from multimeditron.model.data_loader import DataCollatorForMultimodal

    ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
    tokenizer.add_special_tokens(special_tokens)
    attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

    model = MultiModalModelForCausalLM.from_pretrained("path/to/trained/model", device_map="auto")
    model.eval()

    modalities = [{"type" : "image", "value" : "path/to/image"}]
    conversations = [{
        "role" : "user",
            "content" : f"{ATTACHMENT_TOKEN} Describe the image"
    }]
    sample = {
        "conversations" : conversations,
        "modalities" : modalities
    }

    loader = FileSystemImageLoader(base_path=os.getcwd())

    collator = DataCollatorForMultimodal(
            tokenizer=tokenizer,
            tokenizer_type="llama",
            modality_processors=model.processors(), 
            modality_loaders={"image" : loader},
            attachment_token_idx=attachment_token_idx,
            add_generation_prompt=True
    )

    batch = collator([sample])

    with torch.no_grad():
        outputs = model.generate(batch=batch, temperature=0.1)
     
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])

Make sure to adapt the `path/to/trained/model` and the `path/to/image` accordingly.
