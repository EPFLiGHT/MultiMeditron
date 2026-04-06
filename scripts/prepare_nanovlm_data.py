import os
import json
from datasets import load_dataset
from PIL import Image

def main():
    output_dir = "./data/cauldron_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading HuggingFaceM4/FineVision_concat_shuffled_2 (NanoVLM dataset)...")
    # You can remove streaming=True if you want to download the entire multi-GB dataset first
    ds = load_dataset("HuggingFaceM4/FineVision_concat_shuffled_2", split="train", streaming=True)
    
    formatted_data = []
    
    print(f"Formatting the entire 1.7M sample dataset for MultiMeditron...")
    for i, sample in enumerate(ds):
            
        try:
            img = sample['image']
            img_filename = f"image_{i}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            
            # Save the raw image locally 
            if getattr(img, "mode", None) != "RGB":
                img = img.convert("RGB")
            img.save(img_path)
            
            # Reformat Conversations array
            conversations = []
            for turn in sample['texts']:
                conversations.append({
                    "role": turn["role"], 
                    "content": turn["content"].replace("<image>", "<|image|>")
                })
                
            # Create MultiMeditron LLM Training JSON structure
            formatted_data.append({
                "modalities": [
                    {
                        "type": "image",
                        "value": img_filename
                    }
                ],
                "conversations": conversations
            })
        except Exception as e:
            print(f"Skipping sample {i} due to Error: {e}")
            continue

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples...")
            
    # Dump it all natively to JSONL
    output_jsonl = "./data/cauldron_formatted.jsonl"
    with open(output_jsonl, "w") as f:
        for row in formatted_data:
            f.write(json.dumps(row) + "\n")
            
    print(f"Dataset formatting successfully dumped to {output_jsonl}!")

if __name__ == "__main__":
    main()
