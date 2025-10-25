import sys
sys.path.append("../") #to import modules from the rest of the repo
import logging
from src.model.model import MultiModalModelForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset
from itertools import islice
from tqdm import tqdm
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from multimeditron.model.model import ChatTemplate
from src.model.model import MultimodalRawInput
from src.model.prompt_tokenizers import PromptTokenizer
from src.dataset.preprocessor.modality_preprocessor import ModalityRetriever
from src.dataset.registry.fs_registry import FileSystemImageRegistry
from src.model.data_loader import DataCollatorForMultimodal

from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # disables the warning

model_path = "/mloscratch/users/lmartins/openmeditron/MultiMeditron/models/MultiMeditron-8B-CLIP-Alignement-07-August/checkpoint-2508"
llm_path = "meta-llama/Llama-3.1-8B-Instruct"
model_answers_path = "outputs_benchmarks/answers_benchmark_GMAI-MMBench_1306.txt"

ATTACHMENT_TOKEN = "<|reserved_special_token_0|>"

# Load the benchmark dataset
dataset_tsv = "/mloscratch/users/nemo/benchmarking/GMAI-MMBench/GMAI-MMBench_VAL_new.tsv"
dataset_images = "/mloscratch/users/nemo/benchmarking/GMAI-MMBench/images/GMAI-MMBench_VAL/"

dataset = Dataset.from_pandas(pd.read_csv(dataset_tsv, sep="\t", header=0))

batch_size = 16
batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

# Load tokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except:
    logging.warning(f"Loading tokenizer from {llm_path} instead of {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)


tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': [ATTACHMENT_TOKEN]}
tokenizer.add_special_tokens(special_tokens)
attachment_token_idx = tokenizer.convert_tokens_to_ids(ATTACHMENT_TOKEN)

model = MultiModalModelForCausalLM.from_pretrained(model_path)
model.to("cuda")

modalities_num_embeddings = {
    mod_name: processor.num_patches_per_entry for mod_name, processor in model.processors().items()}

prompt_tokenizer = PromptTokenizer(
	tokenizer=tokenizer,
    chat_template=ChatTemplate.from_name("llama"),
	modalities_num_embeddings=modalities_num_embeddings,
	attachment_token_idx=attachment_token_idx
)

if not os.path.exists("outputs_benchmarks"):
    os.mkdir("outputs_benchmarks")

if os.path.exists(model_answers_path):
    with open(model_answers_path, "r") as f:
        answers = list(f.read())
else:
    answers = []

DATA_COLLATOR = DataCollatorForMultimodal(
        tokenizer=tokenizer,
        tokenizer_type="llama",
        modality_processors=model.processors(),
        attachment_token_idx=attachment_token_idx
    )

PROMPT_TEMPLATE = """
"<|reserved_special_token_0|> {question}
{options}
Let's think step by step. Detail your reasoning before answering carefully. Your final answer should have the following format:

<reasoning leading to the answer>
Answer: <letter>

The <letter> is chosen amongst the possible options (A, B, C, D or E)
"""

# Function to process a batch
def process_batch(batch):
    # Prepare modalities for the batch
    modalities_batch = [
        [{"type": "image", "value": os.path.join(dataset_images, f"{index}.jpg")}]
        for index in batch["index"]
    ]

    # Prepare conversations for the batch

    conversations_batch = [
        [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    question=question,
                    options="\n".join(f"{letter}. {opt}" for letter in "ABCDE" if (opt := options.get(letter)) is not None),

                ),
            },
        ]
        for question, options in zip(batch["question"], [
            {letter: batch[letter][i] for letter in "ABCDE"} for i in range(len(batch["question"]))
        ])
    ]

    # Tokenize inputs for the batch
    merged_modalities_batch = []
    for conversations, modalities in zip(conversations_batch, modalities_batch):
        sample = {
            "conversations" : conversations,
            "modalities" : modalities
        }

        modality_retriever = ModalityRetriever(registry=FileSystemImageRegistry(base_path=os.getcwd()))
        sample = modality_retriever.merge_modality_with_sample(sample)
        merged_modalities_batch.append(sample)

    
    batch = DATA_COLLATOR(merged_modalities_batch)
    
    # Generate outputs
    with torch.no_grad():
        outputs_batch = model.generate(
            batch["input_ids"], processed_multimodal_inputs=batch["processed_multimodal_inputs"],
            temperature=0.5, do_sample=True, max_length=1024
        )

    # Decode and extract answers
    batch_answers = []
    with open("outputs_benchmarks/save_1306.txt", "a") as f:
        for i, output in enumerate(outputs_batch):
            rep = tokenizer.decode(output).replace("**", "")
            print(rep)
            f.write(str(i) + ". " + rep + "\n")
                
            try:
                extracted_answer = rep[rep.rindex("Answer: ") + len("Answer: ")][0]
            except:
                extracted_answer = "?"
            batch_answers.append(extracted_answer)

    return batch_answers

# Iterate through the dataset in batches
for i in tqdm(range(len(answers), len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]  # Slice the dataset
    batch_answers = process_batch(batch)
    answers.extend(batch_answers)

    # Save progress after each batch
    with open(model_answers_path, "w") as f:
        f.write("".join(answers))
