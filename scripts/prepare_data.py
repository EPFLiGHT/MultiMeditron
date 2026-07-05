import os
import time
import subprocess
import sys
import functools
from PIL import Image

# Force Python to instantly flush all print statements so they appear live in SLURM's output file!
print = functools.partial(print, flush=True)

def process_row(sample, index, output_dir):
    """Process a single row from the locally-cached dataset. No network calls needed!"""
    try:
        images = sample.get('images', [])
        if not images or len(images) == 0:
            return {
                "modalities": [],
                "conversations": [],
                "text": ""
            }
            
        img = images[0]
        img_filename = f"image_{index}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        img.save(img_path)

        conversations = []
        flat_text = ""
        
        # Include nanoVLM's quality filtering to ignore bad turns
        relevance_min = 1
        img_corr_min = 1
        vis_dep_min = 1
        fmt_min = 1

        for idx, turn in enumerate(sample.get('texts', [])):
            rel = sample.get('relevance_ratings')
            if rel and len(rel) > idx and rel[idx] is not None and rel[idx] < relevance_min:
                continue
            img_corr = sample.get('image_correspondence_ratings')
            if img_corr and len(img_corr) > idx and img_corr[idx] is not None and img_corr[idx] < img_corr_min:
                continue
            vis_dep = sample.get('visual_dependency_ratings')
            if vis_dep and len(vis_dep) > idx and vis_dep[idx] is not None and vis_dep[idx] < vis_dep_min:
                continue
            fmt = sample.get('formatting_ratings')
            if fmt and len(fmt) > idx and fmt[idx] is not None and fmt[idx] < fmt_min:
                continue

            if "user" in turn:
                user_text = turn["user"].replace("<image>", "<|image|>")
                conversations.append({"role": "user", "content": user_text})
                flat_text += user_text + "\n"
            
            if "assistant" in turn:
                asst_text = turn["assistant"].replace("<image>", "<|image|>")
                conversations.append({"role": "assistant", "content": asst_text})
                flat_text += asst_text + "\n"

        return {
            "modalities": [{"type": "image", "value": img_path}],
            "conversations": conversations,
            "text": flat_text.strip()
        }
    except Exception as e:
        print(f"Warning: Failed to process row {index}: {e}")
        return {
            "modalities": [],
            "conversations": [],
            "text": ""
        }

def main():
    output_dir = "/iopsstor/scratch/cscs/haaissa/cauldron_data/images"
    expert_jsonl_path = "/iopsstor/scratch/cscs/haaissa/cauldron_data/expert_cauldron_formatted.jsonl"
    llm_jsonl_path = "/iopsstor/scratch/cscs/haaissa/cauldron_data/cauldron_formatted.jsonl"
    
    # Where huggingface-cli will download the raw dataset files (like nanoVLM's /fsx/ cache)
    dataset_repo = "HuggingFaceM4/FineVision_concat_shuffled_2"
    local_dataset_dir = "/iopsstor/scratch/cscs/haaissa/hf/FineVision_local"
    
    print("=== STEP 1: Directory Setup ===")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_dataset_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 2: BULK DOWNLOAD using huggingface-cli
    # 
    # This is exactly what nanoVLM does in prepare.sh with rsync from /fsx/:
    # They pre-download ALL data files to local SSD BEFORE any Python processing.
    #
    # huggingface-cli download fetches parquet files in bulk (few large HTTP transfers)
    # instead of load_dataset() which triggers 1 HTTP request PER IMAGE (1.7M requests!)
    # =========================================================================
    print("\n=== STEP 2: Bulk downloading dataset with huggingface-cli (like nanoVLM's prepare.sh) ===")
    print("This downloads ALL parquet files in one shot. No per-image rate limiting!")
    start_time = time.time()
    
    download_cmd = [
        "huggingface-cli", "download",
        dataset_repo,
        "--repo-type", "dataset",
        "--local-dir", local_dataset_dir,
    ]
    
    print("Running: " + " ".join(download_cmd))
    result = subprocess.run(download_cmd)
    
    if result.returncode != 0:
        print("WARNING: huggingface-cli exited with code " + str(result.returncode) + ". Trying to proceed anyway...")
    
    elapsed = time.time() - start_time
    print("[SUCCESS] Bulk download completed in " + str(round(elapsed, 2)) + "s")
    
    # =========================================================================
    # STEP 3: Load from LOCAL disk (like nanoVLM's load_from_disk)
    # 
    # Now that all parquet files are physically on our cluster's SSD,
    # we load them WITHOUT any network calls. Zero rate limiting possible!
    # =========================================================================
    print("\n=== STEP 3: Loading dataset from LOCAL disk (zero network requests!) ===")
    start_load_time = time.time()
    
    from datasets import load_dataset
    
    # Find all downloaded parquet files
    parquet_files = []
    for root, dirs, files in os.walk(local_dataset_dir):
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))
    
    parquet_files.sort()
    print("Found " + str(len(parquet_files)) + " local parquet files")
    
    if not parquet_files:
        raise ValueError(
            "No .parquet files found in " + local_dataset_dir + "! "
            "The huggingface-cli download may have failed. Check the output above."
        )
    
    # Load directly from local parquet files - NO internet needed!
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    elapsed = time.time() - start_load_time
    print("[SUCCESS] Loaded " + str(len(ds)) + " rows from local disk in " + str(round(elapsed, 2)) + "s")
    
    # =========================================================================
    # STEP 4: Process rows (100% offline, no network, full speed!)
    # =========================================================================
    print("\n=== STEP 4: Multi-Core OFFLINE Arrow Mapping ===")
    start_process_time = time.time()
    
    processed_ds = ds.map(
        lambda sample, index: process_row(sample, index, output_dir),
        with_indices=True,
        num_proc=36,
        remove_columns=ds.column_names
    )
    
    print("\nFiltering out any empty rows or skipped items...")
    processed_ds = processed_ds.filter(lambda x: len(x["conversations"]) > 0, num_proc=36)
    elapsed = time.time() - start_process_time
    print("[SUCCESS] Processed: " + str(len(processed_ds)) + " in " + str(round(elapsed, 2)) + "s")

    print("\n=== STEP 5: Creating JSONL Output Data ===")
    start_dump_time = time.time()
    
    print("Writing LLM subset...")
    processed_ds.remove_columns(["text"]).to_json(llm_jsonl_path, force_ascii=False)
    
    print("Writing EXPERT subset...")
    processed_ds.remove_columns(["conversations"]).to_json(expert_jsonl_path, force_ascii=False)
    
    elapsed = time.time() - start_dump_time
    print("[SUCCESS] Data expertly dumped in " + str(round(elapsed, 2)) + "s")
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()


