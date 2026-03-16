#!/usr/bin/env python3
"""
Reformat translated datasets for Meditron 4 evaluation
Output format: translation, language, english_text, translation_model
"""

import json
from pathlib import Path
from typing import Optional


# HARDCODED PATHS
INPUT_DIR = "src/multimeditron/translation/datasets/generated_datasets"
OUTPUT_DIR = "src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets"
LANGUAGES = ["amh", "swa", "tam", "hau", "yor", "zul"]
MIN_CONSENSUS_SCORE = None  # Set to e.g., 50.0 to filter low-quality translations


def reformat_for_evaluation(
    input_file: str,
    output_file: str,
    min_consensus_score: Optional[float] = None
):
    """
    Reformat translation output for Meditron 4 evaluation
    
    Args:
        input_file: Path to translated JSONL file
        output_file: Path to save reformatted file
        min_consensus_score: Filter out translations below this score
    """
    
    filtered_count = 0
    total_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line)
            total_count += 1
            
            # Filter by quality if specified
            if min_consensus_score is not None:
                model = entry.get("translation_model", "")
                scores = entry.get("consensus_scores", {})
                score = scores.get(model, 0)
                
                if score < min_consensus_score:
                    filtered_count += 1
                    continue
            
            # Skip empty translations
            if not entry.get("translated_text", "").strip():
                filtered_count += 1
                continue
            
            # Get original English text
            english_text = entry.get("text", entry.get("content", ""))
            
            # Create exact format for Meditron 4
            reformatted = {
                "translation": entry["translated_text"],
                "language": entry["target_language"],
                "english_text": english_text,
                "translation_model": entry.get("translation_model", "")
            }
            
            f_out.write(json.dumps(reformatted, ensure_ascii=False) + '\n')
    
    kept_count = total_count - filtered_count
    print(f"Processed {total_count} entries")
    print(f"Kept: {kept_count}, Filtered: {filtered_count}")
    
    return kept_count


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("REFORMATTING FOR MEDITRON 4 EVALUATION")
    print("Format: translation, language, english_text, translation_model")
    print("="*60)
    print(f"\nInput:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Languages: {', '.join(LANGUAGES)}")
    if MIN_CONSENSUS_SCORE:
        print(f"Min score filter: {MIN_CONSENSUS_SCORE}")
    print()
    
    for lang in LANGUAGES:
        input_file = input_dir / f"{lang}_train.jsonl"
        
        if not input_file.exists():
            print(f"\n⚠ Skipping {lang}: file not found")
            continue
        
        output_file = output_dir / f"{lang}_eval.jsonl"
        
        print(f"\n{lang.upper()}:")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")
        
        kept = reformat_for_evaluation(
            str(input_file),
            str(output_file),
            min_consensus_score=MIN_CONSENSUS_SCORE
        )
        
        print(f"  ✓ Saved {kept} entries")
    
    print("\n" + "="*60)
    print("✓ Reformatting complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()