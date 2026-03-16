"""
Filter large African language dataset (400k+ MCQs) for translation quality issues.

Detects and removes:
1. Untranslated English text
2. Repeated words/phrases
3. Abnormally long words
4. Empty fields

Usage:
    python filter_translated_medibench_finetunedNLLB.py --input large_dataset.json --output clean_dataset.json
"""

import json
import argparse
from collections import Counter
from tqdm import tqdm
from typing import Dict, Tuple


# African language NLLB codes
AFRICAN_LANGUAGES = {
    'am': ('amh_Ethi', 'Amharic'),
    'ha': ('hau_Latn', 'Hausa'),
    'sw': ('swh_Latn', 'Swahili'),
    'yo': ('yor_Latn', 'Yoruba'),
    'zu': ('zul_Latn', 'Zulu'),
}


def detect_translation_issues(text: str, min_length: int = 5) -> Tuple[bool, str]:
    """
    Detect if text has quality issues.
    Returns: (is_valid, reason_if_invalid)
    """
    if not text or len(text.strip()) < min_length:
        return False, "too_short"
    
    words = text.split()
    
    # Check 1: Consecutive repeated words
    if len(words) > 5:
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        if max_consecutive >= 3:
            return False, f"repeated_word_{max_consecutive}x_consecutive"
    
    # Check 2: Word repeated many times throughout text (not just consecutive)
    if len(words) > 10:
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        
        # If any word appears more than 30% of the time, it's suspicious
        if count / len(words) > 0.3:
            return False, f"word_{most_common_word}_repeated_{count}x_total"
    
    # Check 3: Repeated phrases
    if len(text) > 50:
        for phrase_len in range(10, min(30, len(text) // 3)):
            phrase = text[:phrase_len]
            count = text.count(phrase)
            if count >= 3:
                repetition_ratio = (count * phrase_len) / len(text)
                if repetition_ratio > 0.3:
                    return False, f"phrase_repeated_{count}x"
    
    # Check 4: Abnormally long words (concatenation errors)
    max_word_len = max(len(w) for w in words) if words else 0
    if max_word_len > 60:
        return False, f"long_word_{max_word_len}chars"
    
    # Check 5: Too much English (untranslated)
    # Common English words that shouldn't appear in African languages
    english_indicators = [
        'the', 'and', 'is', 'are', 'of', 'to', 'in', 'for', 'a', 'with',
        'patient', 'years', 'old', 'after', 'which', 'following', 'most',
        'treatment', 'diagnosis', 'symptoms', 'blood', 'test', 'examination'
    ]
    
    if len(words) > 10:
        english_count = sum(1 for w in words if w.lower() in english_indicators)
        english_ratio = english_count / len(words)
        
        # If more than 25% of words are common English words, likely untranslated
        if english_ratio > 0.25:
            return False, f"untranslated_english_{english_ratio:.1%}"
    
    # Check 6: Detect full English sentences (stronger check)
    # Pattern: starts with capital letter and has many English words
    if text[0].isupper() and len(words) > 5:
        # Check if first 5 words are all English-like
        first_words = ' '.join(words[:5]).lower()
        english_patterns = [
            'a patient', 'the patient', 'years old', 'male patient',
            'female patient', 'presented with', 'diagnosed with',
            'which of the', 'what is the', 'what are the'
        ]
        
        if any(pattern in first_words for pattern in english_patterns):
            return False, "full_english_sentence"
    
    return True, "valid"


def validate_sample(sample: Dict) -> Tuple[bool, str]:
    """
    Validate entire MCQ sample.
    Returns: (is_valid, reason_if_invalid)
    """
    # Check required fields exist
    question = sample.get('question', '')
    options = sample.get('options', [])
    answer = sample.get('answer', '')
    language = sample.get('language', '')
    
    if not question:
        return False, "missing_question"
    
    if not options or not isinstance(options, list):
        return False, "missing_options"
    
    if len(options) < 2:
        return False, "too_few_options"
    
    if not answer:
        return False, "missing_answer"
    
    if language not in AFRICAN_LANGUAGES:
        return False, f"unknown_language_{language}"
    
    # Validate question quality
    q_valid, q_reason = detect_translation_issues(question, min_length=10)
    if not q_valid:
        return False, f"question_{q_reason}"
    
    # Validate each option
    for i, opt in enumerate(options):
        if not opt or not isinstance(opt, str):
            return False, f"option_{i}_empty"
        
        opt_valid, opt_reason = detect_translation_issues(opt, min_length=2)
        if not opt_valid:
            return False, f"option_{i}_{opt_reason}"
    
    # Validate answer format
    answer_str = str(answer).strip().upper()
    if len(answer_str) != 1 or not answer_str.isalpha():
        return False, f"invalid_answer_format_{answer}"
    
    # Check answer is valid for number of options
    answer_index = ord(answer_str) - ord('A')
    if answer_index < 0 or answer_index >= len(options):
        return False, f"answer_out_of_range_{answer_str}_for_{len(options)}_options"
    
    return True, "valid"


def filter_dataset(input_file: str, output_file: str, stats_file: str = None):
    """
    Filter large dataset for translation quality issues.
    """
    print("="*70)
    print("FILTERING LARGE AFRICAN LANGUAGE DATASET")
    print("="*70)
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   ✅ Loaded {len(data)} samples")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Initialize statistics
    stats = {
        'total': len(data),
        'valid': 0,
        'invalid': 0,
        'by_language': {lang: {'valid': 0, 'invalid': 0} for lang in AFRICAN_LANGUAGES.keys()},
        'rejection_reasons': Counter(),
        'rejected_samples': []  # Store first 100 for review
    }
    
    # Filter samples
    print(f"\n[2/4] Filtering {len(data)} samples...")
    clean_data = []
    
    for sample in tqdm(data, desc="   Filtering"):
        is_valid, reason = validate_sample(sample)
        lang = sample.get('language', 'unknown')
        
        if is_valid:
            clean_data.append(sample)
            stats['valid'] += 1
            if lang in stats['by_language']:
                stats['by_language'][lang]['valid'] += 1
        else:
            stats['invalid'] += 1
            stats['rejection_reasons'][reason] += 1
            
            if lang in stats['by_language']:
                stats['by_language'][lang]['invalid'] += 1
            
            # Store first 100 rejected samples for review
            if len(stats['rejected_samples']) < 100:
                stats['rejected_samples'].append({
                    'language': lang,
                    'reason': reason,
                    'question': sample.get('question', '')[:100],
                    'options': sample.get('options', [])[:2]  # First 2 options
                })
    
    # Calculate percentages
    success_rate = (stats['valid'] / stats['total'] * 100) if stats['total'] > 0 else 0
    
    # Print summary
    print(f"\n[3/4] Filtering Results:")
    print(f"   Total samples: {stats['total']}")
    print(f"   ✅ Valid: {stats['valid']} ({success_rate:.1f}%)")
    print(f"   ❌ Invalid: {stats['invalid']} ({100-success_rate:.1f}%)")
    
    print(f"\n   By language:")
    for lang_code in sorted(AFRICAN_LANGUAGES.keys()):
        lang_name = AFRICAN_LANGUAGES[lang_code][1]
        lang_stats = stats['by_language'][lang_code]
        total = lang_stats['valid'] + lang_stats['invalid']
        rate = (lang_stats['valid'] / total * 100) if total > 0 else 0
        symbol = "✅" if rate >= 85 else "⚠️" if rate >= 70 else "❌"
        print(f"      {symbol} {lang_name:15s} ({lang_code}): {lang_stats['valid']:6d}/{total:6d} ({rate:5.1f}%)")
    
    print(f"\n   Top rejection reasons:")
    for reason, count in stats['rejection_reasons'].most_common(10):
        pct = count / stats['invalid'] * 100 if stats['invalid'] > 0 else 0
        print(f"      {reason:40s}: {count:6d} ({pct:5.1f}%)")
    
    # Save clean dataset
    print(f"\n[4/4] Saving clean dataset...")
    output_path = output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved {len(clean_data)} clean samples to: {output_path}")
    
    # Save statistics
    if not stats_file:
        stats_file = output_file.replace('.json', '_filter_stats.json')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Convert Counter to dict for JSON serialization
        stats_output = {
            **stats,
            'rejection_reasons': dict(stats['rejection_reasons'])
        }
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"   📊 Statistics saved to: {stats_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Success rate: {success_rate:.1f}%")
    print(f"📁 Clean dataset: {len(clean_data)} samples")
    print(f"🗑️  Removed: {stats['invalid']} samples")
    
    if success_rate < 80:
        print(f"\n⚠️  Warning: Success rate is below 80%")
        print(f"   Consider reviewing translation quality or using a different model")
    elif success_rate >= 90:
        print(f"\n✅ Excellent! Success rate is above 90%")
    
    print("="*70)


def quick_stats(input_file: str):
    """
    Quickly check dataset statistics without full filtering.
    """
    print("\n📊 Quick Statistics (sampling 1000 random samples)...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    import random
    sample_size = min(1000, len(data))
    samples = random.sample(data, sample_size)
    
    issues = Counter()
    for sample in tqdm(samples, desc="   Checking"):
        is_valid, reason = validate_sample(sample)
        if not is_valid:
            issues[reason] += 1
    
    valid = sample_size - sum(issues.values())
    print(f"\n   Sampled: {sample_size}")
    print(f"   Valid: {valid} ({valid/sample_size*100:.1f}%)")
    print(f"   Issues found: {sum(issues.values())}")
    
    if issues:
        print(f"\n   Top issues in sample:")
        for reason, count in issues.most_common(5):
            print(f"      {reason}: {count} ({count/sample_size*100:.1f}%)")
    
    # Extrapolate to full dataset
    estimated_valid = int(len(data) * (valid / sample_size))
    estimated_invalid = len(data) - estimated_valid
    
    print(f"\n   📈 Estimated for full dataset:")
    print(f"      Total: {len(data)}")
    print(f"      Valid: ~{estimated_valid}")
    print(f"      Invalid: ~{estimated_invalid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter African language dataset for translation quality issues"
    )
    parser.add_argument(
        "--input",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/mediBench_translation/african_translations.json",
        help="Input JSON file (large dataset)"
    )
    parser.add_argument(
        "--output",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/mediBench_translation/cleaned_high_resource_translation_results_subset.json",
        help="Output JSON file (clean dataset). If not specified, will use input_clean.json"
    )
    parser.add_argument(
        "--stats",
        default="src/multimeditron/translation/experiments/results/finetuned_nllb_consensus/mediBench_translation/cleaned_high_resource_translation_results_stats.json",
        help="Statistics output file. If not specified, will use output_filter_stats.json"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check: sample 1000 items to estimate quality without full filtering"
    )

    args = parser.parse_args()

    if args.quick:
        quick_stats(args.input)
    else:
        filter_dataset(args.input, args.output, args.stats)
