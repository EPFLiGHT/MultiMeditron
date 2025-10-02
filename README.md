# MultiMeditron

MultiMeditron is a multimodal LLM built by students and researchers from [LiGHT lab](https://www.light-laboratory.org/) 

## Setup

To download the project. Execute the following commands:

```
git clone https://github.com/OpenMeditron/MultiMeditron.git
cd MultiMeditron
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install -e .
```

## Inference

To test a model on some modality, you can run the following script. Here is an example for Llama 3.1 8B and a single image:

```py
import torch
from transformers import AutoTokenizer 
import logging
import os

from multimeditron.dataset.preprocessor import modality_preprocessor
from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
from multimeditron.model.model import MultiModalModelForCausalLM 
from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever, SamplePreprocessor
from multimeditron.model.data_loader import DataCollatorForMultimodal

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

model = MultiModalModelForCausalLM.from_pretrained("path/to/trained/model")
model.to("cuda")

modalities = [{"type" : "image", "value" : "path/to/image"}]
conversations = [{
    "role" : "user",
        "content" : f"{ATTACHMENT_TOKEN} Describe the image"
}]
sample = {
    "conversations" : conversations,
    "modalities" : modalities
}

modality_retriever = ModalityRetriever(registry=FileSystemImageRegistry(base_path=os.getcwd()))

collator = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        tokenizer_type="llama",
        modality_processors=model.processors(), 
        attachment_token_idx=attachment_token_idx,
        add_generation_prompt=True
)

batch = collator([sample])

with torch.no_grad():
	outputs = model.generate(batch=batch, temperature=0.1)
 
print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0])
```

## Training

In this part, we describe how we can launch a training of MultiMeditron. Parameters of the training are specified using a `.yaml` configuration file.


### Dataset format

Our training pipeline supports two types of dataset: pretraining and instruction-tuning datasets. Here are the two formats that we support:

1. Arrow/Parquet format (recommended): where the images and modalities are directly stored in the dataset
2. JSONL format: where the images and modalities are stored on the file system. Those dataset must be processed with `merge_inputs.py`

#### Arrow format

1. Pretraining dataset: Each dataset must contain a column `text` and a column `modalities`. The `text` column contains string of the following form:

```py
"Let's compare the first image: <|reserved_special_token_0|>, and the second 3D image: <|reserved_special_token_0|>"
```
And the `modalities` column must be of the following form:

```py
[{"type": "modality_type", "value" : some_modality}]
```

For instance, for image type, `some_modality` must contains the bytes of the image

Note that we use a special placeholder `<|reserved_special_token_0|>` to indicate the position of the tokens from the modality

2. Instruction-tuning dataset: It's the same as the pretraining dataset but instead of the `text` column, we have a `conversations` column:

```py
[
    {"role" : "system", "content" : "You are Meditron"},
    {"role" : "user", "content" : "Compare the CT scan <|reserved_special_token_0|> with the image <|reserved_special_token_0|>."},
    {"role" : "assistant", "content" : "Lorem ipsum dolor sit amet, consectetur adipiscing elit."},
    {"role" : "user", "content" : "How is it related to that signal: <|reserved_special_token_0|>?"},
    {"role" : "assistant", "content" : "Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor."}
]
```

#### JSONL format

We also support `.jsonl` files where each line is a sample. We describe how each sample must be formatted:

1. Pretraining format:
```json
{
  "text": "Let's compare the first image: <|reserved_special_token_0|>, and the second 3D image: <|reserved_special_token_0|>",
  "modalities": [{"type" : "image", "value" : "path/to/png"}, {"type" : "image_3d", "value" : "path/to/npy"}]
}
```

2. Instruction-tuning format:
```json
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
```

The instruction-tuning format follows a similar format.

### Merging modalities with the dataset

If you have your dataset in a JSONL format, you can convert it to the parquet/arrow format by running the following command:
```
python merge_inputs -c path/to/config.yaml
```

Where the `config.yaml` file is described later.


### Training

To train the model, we provide a script to launch the training. The script is located in `train_alignment.py`. To run it, use the following command:

```bash
torchrun --nproc-per-node $PROC_PER_NODE -m multimeditron train --config path/to/config.yaml
```

The configuration file must contain the following parameters:

```yaml
base_llm: # (str) Path to LLM model (can be a local model or a model stored on huggingface)
base_model: # (str) Path to trained model. If empty, the LLM model will be initialized to the weights of base_llm, the CLIP are initialized to their default values and projections are initialized randomly
attachment_token: # (str) Attachment placeholder in the prompts. Should be <|reserved_special_token_0|>
tokenizer_type: # (str) The type of tokenizer that should be used, depends on the model (supported values are llama and apertus)
token_size: # (int) Dimension of the embedding of a token for the LLM
truncation: # (Optional[boolean]) Whether to truncate the input or not, default to false
max_sequence_length: # (Optional[int]) The maximum sequence length if truncation is enabled

modalities:
    model_type: # (str) Type of the modality used (supported value are meditron_clip, meditron_pe or moe_meditron_clip)
    config: # (Dict[str, str]) Configuration passed to the modality

training_mode: # (str) Either ALIGNMENT, END2END or FULL. If ALIGNMENT, this will train the projection layer while freezing every other weights. If END2END, this will train the LLM+Projection while freezing every other weights. If FULL, this will train all the model at the same time

datasets: # List of datasets to use for finetuning. Each dataset must follow the format described in the README.md
  - packed_path: # (str) Path to the 1st dataset
  - packed_path: # (str) Path to the 2nd dataset

training_args: # Huggingface training arguments. Check the following documentation for more informations: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments


