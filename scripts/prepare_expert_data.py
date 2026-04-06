import os
import json
from datasets import load_dataset
from PIL import Image

def main():
    output_dir = "./data/cauldron_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading HuggingFaceM4/FineVision_concat_shuffled_2 (NanoVLM dataset)...")
    ds = load_dataset("HuggingFaceM4/FineVision_concat_shuffled_2", split="train", streaming=True)
    
    expert_data = []
    
    print(f"Formatting Phase 1 (Expert) dataset for MultiMeditron...")
    for i, sample in enumerate(ds):
            
        try:
            img = sample['image']
            img_filename = f"image_{i}.jpg"
            img_path = os.path.join(output_dir, img_filename)
            
            # Save the raw image locally 
            if getattr(img, "mode", None) != "RGB":
                img = img.convert("RGB")
            # Only save once. If you already ran the other script, you can comment this out
            img.save(img_path) 
            
            # Phase 1 Expert Format requires FLAT TEXT, not a conversation array
            # We compress the dialogue into a single string
            flat_text = ""
            for turn in sample['texts']:
                # The question
                if turn["role"] == "user":
                    flat_text += turn["content"].replace("<image>", "<|image|>") + "\n"
                # The answer
                elif turn["role"] == "assistant":
                    flat_text += turn["content"] + "\n"
                
            # Create MultiMeditron Expert Training JSON structure
            expert_data.append({
                "modalities": [
                    {
                        "type": "image",
                        "value": img_filename
                    }
                ],
                "text": flat_text.strip() # The "Simple" key the expert requires
            })
        except Exception as e:
            continue

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples...")
            
    # Dump it perfectly to JSONL
    output_jsonl = "./data/expert_cauldron_formatted.jsonl"
    with open(output_jsonl, "w") as f:
        for row in expert_data:
            f.write(json.dumps(row) + "\n")
            
    print(f"Expert Dataset formatting successfully dumped to {output_jsonl}!")

if __name__ == "__main__":
    main()
