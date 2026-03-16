"""
Hindi and Punjabi Healthcare Dataset Converter

Converts Hindi and Punjabi healthcare CSV datasets into Meditron pretraining JSONL format.
Creates natural language medical case texts from structured patient records including
diagnosis, patient history, symptoms, treatment, and demographics.

Input: CSV files with medical patient records
Output: JSONL files with {"text": "...", "modalities": []}

Usage:
    python convert_healthcare_to_pretraining.py
"""

import os
import json
import pandas as pd


INPUT_FILES = {
    'hindi': 'src/multimeditron/translation/datasets/raw/healthcare_hindi_punjabi/hindi_dataset.csv',
    'punjabi': 'src/multimeditron/translation/datasets/raw/healthcare_hindi_punjabi/punjabi_dataset.csv'
}

OUTPUT_DIR = 'src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/healthcare_hindi_punjabi'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_language(row):
    """Detect language from gender field text content."""
    gender = str(row.get('gender', '')).strip()
    if 'पुरुष' in gender or 'महिला' in gender:
        return 'Hindi'
    elif 'ਮਰਦ' in gender or 'ਔਰਤ' in gender:
        return 'Punjabi'
    return 'Unknown'


def create_pretraining_text(row):
    """
    Create natural medical case text from patient record.
    Combines diagnosis, patient history, symptoms, treatment, etc.
    """
    language = detect_language(row)
    text_parts = []
    
    diagnosis = str(row.get('Diagnosis', '')).strip() if pd.notna(row.get('Diagnosis')) else ''
    patient_history = str(row.get('Patient History', '')).strip() if pd.notna(row.get('Patient History')) else ''
    symptoms = str(row.get('symptoms', '')).strip() if pd.notna(row.get('symptoms')) else ''
    treatment = str(row.get('treatment', '')).strip() if pd.notna(row.get('treatment')) else ''
    remarks = str(row.get('Remarks', '')).strip() if pd.notna(row.get('Remarks')) else ''
    timespan = str(row.get('timespan', '')).strip() if pd.notna(row.get('timespan')) else ''
    diagnosis_category = str(row.get('Diagnosis Category', '')).strip() if pd.notna(row.get('Diagnosis Category')) else ''
    age = str(row.get('age', '')).strip() if pd.notna(row.get('age')) else ''
    gender = str(row.get('gender', '')).strip() if pd.notna(row.get('gender')) else ''
    
    if diagnosis:
        if language == 'Hindi':
            text_parts.append(f"निदान: {diagnosis}")
        elif language == 'Punjabi':
            text_parts.append(f"ਨਿਦਾਨ: {diagnosis}")
        else:
            text_parts.append(f"Diagnosis: {diagnosis}")
    
    if age and gender:
        if language == 'Hindi':
            text_parts.append(f"रोगी विवरण: {age} वर्ष, {gender}")
        elif language == 'Punjabi':
            text_parts.append(f"ਮਰੀਜ਼ ਵੇਰਵਾ: {age} ਸਾਲ, {gender}")
        else:
            text_parts.append(f"Patient: {age} years, {gender}")
    
    if patient_history:
        if language == 'Hindi':
            text_parts.append(f"रोगी का इतिहास: {patient_history}")
        elif language == 'Punjabi':
            text_parts.append(f"ਮਰੀਜ਼ ਦਾ ਇਤਿਹਾਸ: {patient_history}")
        else:
            text_parts.append(f"Patient History: {patient_history}")
    
    if symptoms:
        if language == 'Hindi':
            text_parts.append(f"लक्षण: {symptoms}")
        elif language == 'Punjabi':
            text_parts.append(f"ਲੱਛਣ: {symptoms}")
        else:
            text_parts.append(f"Symptoms: {symptoms}")
    
    if treatment:
        if language == 'Hindi':
            text_parts.append(f"उपचार: {treatment}")
        elif language == 'Punjabi':
            text_parts.append(f"ਇਲਾਜ: {treatment}")
        else:
            text_parts.append(f"Treatment: {treatment}")
    
    if timespan:
        if language == 'Hindi':
            text_parts.append(f"उपचार अवधि: {timespan}")
        elif language == 'Punjabi':
            text_parts.append(f"ਇਲਾਜ ਦੀ ਮਿਆਦ: {timespan}")
        else:
            text_parts.append(f"Treatment Timeline: {timespan}")
    
    if remarks:
        if language == 'Hindi':
            text_parts.append(f"टिप्पणी: {remarks}")
        elif language == 'Punjabi':
            text_parts.append(f"ਟਿੱਪਣੀ: {remarks}")
        else:
            text_parts.append(f"Remarks: {remarks}")
    
    if diagnosis_category:
        if language == 'Hindi':
            text_parts.append(f"श्रेणी: {diagnosis_category}")
        elif language == 'Punjabi':
            text_parts.append(f"ਸ਼੍ਰੇਣੀ: {diagnosis_category}")
        else:
            text_parts.append(f"Category: {diagnosis_category}")
    
    full_text = " ".join(text_parts)
    return full_text.strip()


def convert_csv_to_jsonl(csv_path, output_path, language):
    """Convert CSV to pretraining JSONL format."""
    print(f"\n🔄 Processing {language} dataset: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {list(df.columns)}")
    
    valid_count = 0
    skipped_count = 0
    total_chars = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            text = create_pretraining_text(row)
            
            if not text or len(text.strip()) < 50:
                skipped_count += 1
                continue
            
            entry = {
                "text": text,
                "modalities": []
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            valid_count += 1
            total_chars += len(text)
    
    avg_length = total_chars / valid_count if valid_count > 0 else 0
    print(f"   ✅ Wrote {valid_count} examples")
    print(f"   📏 Average text length: {avg_length:.0f} characters")
    print(f"   ⚠️  Skipped {skipped_count} examples (too short)")
    print(f"   📁 Output: {output_path}")
    
    return valid_count, skipped_count


def main():
    print("="*80)
    print("🏥 Healthcare Dataset → Pretraining JSONL Converter")
    print("="*80)
    
    total_valid = 0
    total_skipped = 0
    
    for language, input_file in INPUT_FILES.items():
        output_file = os.path.join(OUTPUT_DIR, f'healthcare_{language}_pretraining.jsonl')
        
        if not os.path.exists(input_file):
            print(f"\n❌ File not found: {input_file}")
            print(f"   Please ensure the file is in the correct location.")
            continue
        
        valid, skipped = convert_csv_to_jsonl(input_file, output_file, language)
        total_valid += valid
        total_skipped += skipped
    
    print("\n" + "="*80)
    print("📊 CONVERSION SUMMARY")
    print("="*80)
    print(f"✅ Total examples created: {total_valid}")
    print(f"⚠️  Total examples skipped: {total_skipped}")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print("\n✨ Conversion complete!")


if __name__ == "__main__":
    main()