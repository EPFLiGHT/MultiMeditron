"""Code snippet for inference"""

import torch
from transformers import AutoTokenizer 
import logging
import os

from multimeditron.model.model import MultiModalModelForCausalLM 
from multimeditron.model.data_loader import DataCollatorForMultimodal
from multimeditron.dataset.loader import FileSystemImageLoader
import argparse

default_model = "ClosedMeditron/Mulimeditron-Proj-CLIP-generalist"
default_llm = "meta-llama/Llama-3.1-8B-Instruct"

parser = argparse.ArgumentParser(description="Example to run inference on Meditron")
parser.add_argument("--model_checkpoint", required=False, default=default_model)
args = parser.parse_args()

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

model_name = args.model_checkpoint

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
except:
    logging.warning(f"Loading tokenizer from {default_llm}")
    tokenizer = AutoTokenizer.from_pretrained(default_llm, padding_side="left")

tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

model = MultiModalModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, use_safetensors=True)
model.to("cuda")

print(model)

modalities1 = [dict(
    type="image",
    value="mock_dataset/forearm_schema.jpg",
)]
modalities2 = [dict(
    type="image",
    value="mock_dataset/infarcted_brain.jpg",
)]

conversations1 = [{
    "role": "user",
    "content": f"{ATTACHMENT_TOKEN} List all the muscles in the image"
}]
conversations2 = [{
    "role": "user",
    "content": f"{ATTACHMENT_TOKEN} What is your diagnosis?"
}]
conversations3 = [{
    "role": "user",
    "content": f"Hello!"
}]


sample1 = {
    "conversations" : conversations1,
    "modalities": modalities1
}

sample2 = {
    "conversations" : conversations2,
    "modalities": modalities2
}

sample3 = {
    "conversations" : conversations3,
    "modalities": []
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

batch = collator([sample1, sample2, sample3])

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model.generate(batch=batch, 
                             temperature=0.7, do_sample=True, max_new_tokens=512)

res = tokenizer.batch_decode(
    outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

for output in res:
    print(output)
    print("=" * 50)


