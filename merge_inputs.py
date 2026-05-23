from datasets import Dataset
from multimeditron.dataset.preprocessor.modality_preprocessor import ModalityRetriever

from multimeditron.dataset.registry.registry import get_registry 
import torch
from multimeditron.model.jsonl_generator import JSONLGenerator
import argparse
import yaml
import torch
from transformers import AutoTokenizer
import torch.multiprocessing as mp


torch.set_num_threads(1)
mp.set_sharing_strategy("file_system")
mp.set_start_method("spawn", force=True)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Path of the MultiMeditron configuration")
    args = parser.parse_args()

    # Load the configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    ATTACHMENT_TOKEN = config["attachment_token"]

    # Disable randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create the base model with fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_llm"], dtype=torch.bfloat16, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens_list = [ATTACHMENT_TOKEN, "<|global_image|>"]
    for i in range(1, 9):
        for j in range(1, 9):
            special_tokens_list.append(f"<row_{i}_col_{j}>")
    special_tokens = {'additional_special_tokens': special_tokens_list}
    tokenizer.add_special_tokens(special_tokens)

    # Create a model
    torch.set_default_dtype(torch.bfloat16)

    for config_dataset in config["datasets"]:
        ds_name = config_dataset["tokenized_path"]

        print(f"Saving dataset to {ds_name}")

        # Retrieve the registry type
        registry_builder = get_registry(
            config_dataset.get("registry_type", "path"))

        with registry_builder(**config_dataset["preprocessor_kwargs"]) as registry:
            num_processes = config_dataset.get("num_processes", 32)

            # Tokenize the inputs
            ds = JSONLGenerator(config_dataset["jsonl_path"])
            ds = Dataset.from_generator(lambda: ds)

            if "num_shards" in config_dataset and "shard_id" in config_dataset:
                print(f"Sharding with num_shards ({config_dataset['num_shards']}) \
                        with shard_id ({config_dataset['shard_id']})")
                ds: Dataset = ds.shard(
                        index=config_dataset["shard_id"],
                        num_shards=config_dataset["num_shards"]
                    )

            # Filter the inputs
            if config_dataset.get("check_input", False):
                ds = ds.filter(registry.check_sample,
                               batched=False, num_proc=num_processes,
                               writer_batch_size=num_processes,
                               desc="Filtering samples")

            processor_wrapper = ModalityRetriever(registry)

            ds = ds.map(processor_wrapper.merge_modality_with_sample,
                        batched=False,
                        writer_batch_size=num_processes, 
                        num_proc=num_processes)

            ds.save_to_disk(f"{ds_name}", num_proc=num_processes)

if __name__ == "__main__":
    main()

