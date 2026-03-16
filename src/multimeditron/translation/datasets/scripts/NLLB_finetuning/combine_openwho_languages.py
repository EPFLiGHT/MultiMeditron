"""
Combine OpenWHO Language Datasets

Combines all 48 OpenWHO language pair datasets into a single multilingual dataset
for NLLB fine-tuning. Each example in the combined dataset contains exactly 2 languages:
English (eng_Latn) and one target language.

The script manually creates examples as dictionaries to avoid schema unification issues
that can occur when concatenating datasets with different language keys. This ensures
consistent data structure across all language pairs.

Input: Individual language pair datasets in src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_formatted/
Output: Single combined dataset in src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_all_languages/

Usage:
    python combine_openwho_languages.py
    python combine_openwho_languages.py --input_dir <path> --output_dir <path>
"""

from datasets import load_from_disk, Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def combine_all_languages(
    input_dir: str = "src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_formatted", 
    output_dir: str = "src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_all_languages"
):
    """
    Combine all prepared language datasets into one multilingual dataset.
    Manually creates examples to avoid schema unification issues.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    print("="*70)
    print("Combining All OpenWHO Languages")
    print("="*70)
    
    lang_dirs = sorted([d for d in input_path.iterdir() 
                       if d.is_dir() and d.name.startswith('eng_Latn-')])
    
    print(f"\nFound {len(lang_dirs)} language pairs:")
    for d in lang_dirs[:5]:
        print(f"  • {d.name}")
    if len(lang_dirs) > 5:
        print(f"  ... and {len(lang_dirs) - 5} more")
    
    all_examples = []
    lang_counts = Counter()
    
    print("\nLoading datasets...")
    for lang_dir in tqdm(lang_dirs, desc="Loading"):
        try:
            dataset = load_from_disk(str(lang_dir))
            
            if 'train' not in dataset:
                print(f"\n  Warning: No 'train' split in {lang_dir.name}")
                continue
            
            train_data = dataset['train']
            target_lang = lang_dir.name.split('-')[1]
            lang_counts[target_lang] += len(train_data)
            
            for example in train_data:
                new_example = {
                    'translation': {
                        'eng_Latn': example['translation']['eng_Latn'],
                        target_lang: example['translation'][target_lang]
                    },
                    'course_id': example.get('course_id', ''),
                    'section_index': example.get('section_index', 0),
                    'subsection_index': example.get('subsection_index', 0)
                }
                all_examples.append(new_example)
            
            if len(all_examples) == len(train_data):
                ex = all_examples[0]
                langs = list(ex['translation'].keys())
                print(f"\n  Example from {lang_dir.name}:")
                print(f"    Languages in example: {langs}")
                print(f"    Number of keys: {len(langs)}")
                print(f"    {langs[0]}: {ex['translation'][langs[0]][:60]}...")
                print(f"    {langs[1]}: {ex['translation'][langs[1]][:60]}...")
                
        except Exception as e:
            print(f"\n  Error loading {lang_dir.name}: {e}")
    
    print(f"\n✓ Loaded {len(lang_dirs)} datasets")
    print(f"✓ Total examples collected: {len(all_examples):,}")
    
    print("\nCreating combined dataset...")
    
    translations = [ex['translation'] for ex in all_examples]
    course_ids = [ex['course_id'] for ex in all_examples]
    section_indices = [ex['section_index'] for ex in all_examples]
    subsection_indices = [ex['subsection_index'] for ex in all_examples]
    
    combined_train = Dataset.from_dict({
        'translation': translations,
        'course_id': course_ids,
        'section_index': section_indices,
        'subsection_index': subsection_indices
    })
    
    print("\nVerifying data structure...")
    sample = combined_train[0]
    sample_langs = list(sample['translation'].keys())
    print(f"  Sample example languages: {sample_langs}")
    print(f"  Number of languages per example: {len(sample_langs)}")
    
    if len(sample_langs) == 2:
        print("  ✓ Each example has exactly 2 languages")
    else:
        print(f"  ✗ ERROR! Expected 2 languages, got {len(sample_langs)}")
    
    none_count = sum(1 for val in sample['translation'].values() if val is None)
    if none_count == 0:
        print("  ✓ No None values in sample")
    else:
        print(f"  ✗ WARNING: Found {none_count} None values")
    
    print("\n  Checking multiple examples...")
    all_good = True
    for i in range(min(100, len(combined_train))):
        ex = combined_train[i]
        if len(ex['translation'].keys()) != 2:
            print(f"    ✗ Example {i} has {len(ex['translation'].keys())} languages!")
            all_good = False
            break
        for val in ex['translation'].values():
            if val is None:
                print(f"    ✗ Example {i} has None value!")
                all_good = False
                break
    
    if all_good:
        print("  ✓ Verified 100 examples - all have 2 languages, no None values")
    
    combined_dataset = DatasetDict({
        'train': combined_train
    })
    
    print(f"\nSaving combined dataset to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    combined_dataset.save_to_disk(str(output_path))
    
    print("\n" + "="*70)
    print("✓ Combined Dataset Created")
    print("="*70)
    print(f"  Location: {output_path}")
    print(f"  Languages: {len(lang_counts)}")
    print(f"  Total examples: {len(combined_train):,}")
    print(f"  Size on disk: ~{sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    
    print("\n" + "="*70)
    print("Language Distribution:")
    print("="*70)
    
    sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    
    for lang, count in sorted_langs[:10]:
        percentage = count / len(all_examples) * 100
        print(f"  {lang:20} {count:5,} examples ({percentage:5.2f}%)")
    
    if len(sorted_langs) > 10:
        print(f"  ... and {len(sorted_langs) - 10} more languages")
    
    print("\n" + "="*70)
    print("Next Step: Train the model!")
    print("="*70)
    print("\nRun: python src/multimeditron/translation/datasets/training/finetune_nllb.py")
    
    return combined_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine all OpenWHO languages")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_formatted",
        help="Directory containing individual language datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/openwho_all_languages",
        help="Where to save combined dataset"
    )
    
    args = parser.parse_args()
    
    combined_dataset = combine_all_languages(args.input_dir, args.output_dir)
    
    print("\n✅ Dataset ready for training!")