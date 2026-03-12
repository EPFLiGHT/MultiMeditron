"""
AfriBERTa Medical Corpus Filtering

Filters the AfriBERTa corpus to extract medical content for pretraining.
Uses strict filtering requiring multiple medical keywords per line to ensure
high-quality medical content while excluding religious, political, cultural,
sports, entertainment, and metadata content.

The script processes 10 African languages:
- Afaan Oromoo, Amharic, Gahuza, Hausa, Igbo
- Pidgin, Somali, Swahili, Tigrinya, Yoruba

Input: AfriBERTa corpus from HuggingFace (castorini/afriberta-corpus)
Output: Filtered medical JSONL files in datasets/formatted_datasets/healthcare_datasets/afriberta_medical_jsonl/

Usage:
    python convert_afriBERT_to_pretraining.py
"""

import os
import json
import zipfile
import unicodedata
import requests
from io import BytesIO


RAW_DIR = "src/multimeditron/translation/datasets/raw/afriberta_raw/"
OUT_DIR = "src/multimeditron/translation/datasets/formatted_datasets/healthcare_datasets/afriberta_medical_jsonl/"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES = [
    "afaanoromoo", "amharic", "gahuza", "hausa", "igbo",
    "pidgin", "somali", "swahili", "tigrinya", "yoruba"
]


BAD_CONTEXT = [
    # Swahili
    "yesu", "mungu", "kanisa", "biblia", "waziri", "rais", "serikali", "siasa", "uhuru",
    "dini", "islami", "kristo", "imani", "maombi", "sheikh", "padri", "nabii", "wanasiasa",
    "utamaduni", "mfalme", "malkia", "uchaguzi", "chama", "upinzani", "bunge", "seneta",
    "gazeti", "redio", "televisheni", "mchezo", "muziki", "sinema", "filamu", "hadithi",
    "mashairi", "wimbo", "ngoma", "sanaa", "jamii", "familia", "ndoa", "harusi", "biashara",
    
    # Yoruba
    "jesu", "olorun", "ijo", "bibile", "ijoba", "alufa", "alfa", "imoletan", "awon omo ogun",
    "ijo kristeni", "oselu", "oba", "ayeye", "iran", "itan", "adamo", "iman", "oloriburuku",
    "idibo", "egbe", "aṣoju", "minista", "iwe irohin", "ere", "orin", "fiimu", "itan-akọọlẹ",
    
    # Hausa
    "yesu", "allah", "coci", "addini", "musulmi", "kirista", "bishiya", "ubangiji", "malam",
    "liman", "majalisar", "gwamnati", "jam'iyya", "siyasa", "sarki", "masarauta", "bature",
    "zabe", "minista", "jarida", "wasa", "waka", "fina-finai", "labari", "tarihi",
    
    # Amharic
    "ኢየሱስ", "ክርስቶስ", "አላህ", "አምላክ", "መንግስት", "ፕሬዝዳንት", "መንፈስ", "ቤተክርስቲያን",
    "ኢስላም", "ክርስትና", "እምነት", "ጸሎት", "ሙስሊም", "ታሪክ", "አስተዳደር", "ምርጫ", "ፓርቲ",
    "ጋዜጣ", "ሬድዮ", "ቴሌቪዥን", "ጨዋታ", "ሙዚቃ", "ፊልም", "ስፖርት",
    
    # Afaan Oromoo
    "yesuus", "waaqayyo", "ammantii", "mootummaa", "presidantii", "mana kiristaanaa", 
    "masgiida", "kadhannaa", "sirna", "aadaa", "ummata", "raayyaa", "mirga namoomaa", 
    "poolisii", "oromiyaa", "waajjira", "bilisummaa", "mormii", "sirna mootummaa", 
    "guddina siyaasaa", "salaata", "masgida", "filannoo", "paartii", "gaazexaa", 
    "taphataa", "muuziqaa", "ispoortii", "seenaa",
    
    # Somali
    "ciise", "ilaah", "masaajid", "kaniisad", "diin", "daacad", "madaxwayne", "xukuumad",
    "dadweyne", "ciidan", "dhaqan", "taariikh", "boqor", "qoyska", "hees", "kaniisada",
    "doorasho", "xisbi", "wasiir", "jariirad", "ciyaar", "heeso", "film", "isboort",
    
    # Igbo
    "jesu", "chukwu", "nna ukwu", "akwukwo nso", "ndị ọchịchị", "ndị isi ala",
    "ndị ụka", "ọbá", "omenala", "egwuregwu", "ncheta", "ntuli aka", "otu ndọrọndọrọ",
    "akwụkwọ akụkọ", "egwuregwu", "egwu", "ihe nkiri", "akụkọ",
    
    # Gahuza / Kinyarwanda
    "yesu", "imana", "idini", "kiliziya", "pasiteri", "musilimu", "abaperezida",
    "leta", "ubutegetsi", "umwami", "umuco", "amateka", "abayobozi", "amatora",
    "ishyaka", "ikinyamakuru", "umukino", "indirimbo", "filime", "siporo",
    
    # Tigrinya
    "ኢየሱስ", "ክርስቶስ", "አላህ", "አምላክ", "መንግስቲ", "መንፈስ ቅዱስ", "ቤተ ክርስቲያን",
    "እስልምና", "ፕሬዝዳንት", "ሙስሊም", "ታሪኽ", "መሳርሕ", "ምርጫ", "ፓርቲ", "ጋዜጣ",
    "ጸወታ", "ሙዚቃ", "ፊልም", "ስፖርት",
    
    # Pidgin
    "jesus", "god", "church", "bible", "goment", "president", "minister", "politics",
    "election", "party", "newspaper", "game", "music", "movie", "sport", "pastor",
    
    # General patterns
    "video", "music", "movie", "sport", "football", "election", "president", "minister",
    "government", "party", "church", "mosque", "prayer", "bible", "quran", "allah",
]

METADATA_PATTERNS = [
    "bbc", "news", "iroyin", "labaran", "berita", "breaking", "update",
    "source:", "photo:", "image:", "video:", "duration", "previous article",
    "next article", "related", "share", "comment", "like", "subscribe",
    "http", "www", ".com", "click", "link", "download", "mp3", "mp4",
    "wey don pass", "hours wey don pass", "akójọpọ̀ iroyin",
]


def looks_medical(line: str) -> bool:
    """
    Check if line passes contextual filter by excluding religious, political,
    cultural, sports, entertainment, metadata, and navigation content.
    """
    line_lower = line.lower()
    
    for bad in BAD_CONTEXT:
        if bad in line_lower:
            return False
    
    for pattern in METADATA_PATTERNS:
        if pattern in line_lower:
            return False
    
    return True


MEDICAL_KEYWORDS = {
    "swahili": [
        "hospitali", "daktari", "mgonjwa", "kliniki", "wodi", "daktari wa meno", 
        "muuguzi", "tabibu", "ugonjwa", "virusi", "maambukizi", "corona", "covid", 
        "malaria", "ukimwi", "homa", "ebola", "kansa", "sukari", "presha", "vidonda", 
        "mafua", "kifua kikuu", "moyo", "mapafu", "damu", "figo", "ini", "tezi", 
        "mishipa", "neva", "tiba", "dawa", "matibabu", "chanjo", "uchunguzi", 
        "upasuaji", "maabara", "mimba", "uzazi", "ujauzito", "mtoto mchanga",
    ],
    
    "amharic": [
        "ሆስፒታል", "ሐኪም", "ክሊኒክ", "ታካሚ", "ዶክተር", "ነርስ", "ቫይረስ", "በሽታ", 
        "ኢንፌክሽን", "ኮቪድ", "ኤይድስ", "ኢቦላ", "ማለሪያ", "ካንሰር", "ደም", "ልብ", 
        "ጉበት", "ኩላሊት", "አንጀት", "ሆድ", "መድሀኒት", "ሕክምና", "ክትባት", "መርመር", 
        "ወሊድ", "ሕፃን", "እናት",
    ],
    
    "afaanoromoo": [
        "hospitaala", "kilinika", "doktora", "dhukkuba", "dhibee", "vaayirasii", 
        "maleriya", "hiv", "aids", "ebola", "kansara", "dhukkuba qorraa", "onnee", 
        "dhiiga", "somaa", "ulfa", "yaala", "daaw'aa", "fayyaa", "talaallii", 
        "qoricha", "daa'ima", "haadha", "ulfaa",
    ],
    
    "yoruba": [
        "ile-iwosan", "dokita", "ile-egboogi", "aisan", "arun", "maleria", "covid", 
        "ebola", "àtọgbẹ", "arun ọkan", "arun ẹdọ", "arun kidinrin", "ẹjẹ", "ọkàn", 
        "ẹdọ", "kidinrin", "ẹdọforo", "ọpọlọ", "oogun", "iwosan", "ilera", "ajẹsara", 
        "aboyun", "ọmọ", "ilera obinrin",
    ],
    
    "somali": [
        "isbitaalka", "dhaqtarka", "bukaanka", "qalliinka", "cudurka", "malariya", 
        "aids", "covid", "ebola", "wadne xanuun", "kansar", "sonkorow", "qandho", 
        "wadne", "beerka", "kelyaha", "dhiig", "maskax", "daawo", "caafimaad", 
        "tallaalka", "baaritaan", "ilmo", "dhalmo", "hooyo",
    ],
    
    "hausa": [
        "asibiti", "likita", "majinyaci", "jinya", "asibitin mata", "cutar", "maleriya", 
        "aids", "hiv", "ebola", "ciwon zuciya", "ciwon daji", "ciwon huhu", "ciwon suga", 
        "jini", "hanta", "koda", "ido", "kai", "ciki", "magani", "rigakafi", "allura", 
        "magunguna", "haihuwa", "mata masu juna biyu", "jariri",
    ],
    
    "pidgin": [
        "doctor", "hospital", "nurse", "clinic", "malaria", "covid", "cholera", 
        "diabetes", "pressure", "heart disease", "ebola", "fever", "infection", 
        "cancer", "blood", "heart", "eye", "ear", "tooth", "skin", "lung", "kidney", 
        "liver", "medicine", "checkup", "operation", "pill", "injection", "pregnant", 
        "belle", "baby", "midwife",
    ],
    
    "igbo": [
        "dọkịta", "ụlọ ọgwụ", "ụlọ ọgwụ ụmụaka", "ọrịa", "ọrịa shuga", "ọrịa obi", 
        "ọrịa ụbụrụ", "ọrịa ara", "ọrịa nsị", "ọrịa kansa", "ọbara", "obi", "ọkpụkpụ", 
        "ahụ", "anya", "ntị", "ọgwụ", "ahụike", "ọgwụgwọ", "nwaanyị ime", "ụmụaka", 
        "ụlọ ọmụmụ",
    ],
    
    "gahuza": [
        "ibitaro", "umuganga", "kilinika", "indwara", "sida", "malariya", "corona", 
        "ibibari", "indwara y'umutima", "kanseri", "amaraso", "umutima", "uruhago", 
        "inkari", "amaso", "ubuzima", "ubuvuzi", "imiti", "inkingo", "ababyeyi", 
        "umwana", "kubyara",
    ],
    
    "tigrinya": [
        "ሆስፒታል", "ሓኪም", "ክሊኒክ", "ታካሚ", "ቫይረስ", "በሽታ", "ኮቪድ", "ኤይድስ", 
        "ማለሪያ", "ኤቦላ", "ካንሰር", "ደም", "ልብ", "ጉበት", "ኩላሊት", "መድሃኒት", 
        "ሕክምና", "ክትባት", "ወሊድ", "እናት", "ሕፃን",
    ]
}


def download_language(lang: str):
    """Download and extract AfriBERTa corpus for a specific language."""
    out_dir = os.path.join(RAW_DIR, lang)
    train_path = os.path.join(out_dir, "train.txt")
    if os.path.exists(train_path):
        print(f"⚡ {lang}: already downloaded, skipping.")
        return train_path

    url = f"https://huggingface.co/datasets/castorini/afriberta-corpus/resolve/main/{lang}/train.zip"
    os.makedirs(out_dir, exist_ok=True)

    print(f"⬇️  Downloading {lang}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(BytesIO(r.content)) as z:
            z.extractall(out_dir)
        print(f"✅ Extracted {lang} to {out_dir}")
        return train_path
    except Exception as e:
        print(f"❌ Failed to download {lang}: {e}")
        return None


def filter_medical_lines(text_path: str, lang: str, min_keywords: int = 2):
    """
    Filter for medical content by requiring:
    1. At least min_keywords medical terms (default: 2)
    2. Minimum line length
    3. Pass contextual filter
    """
    keywords = MEDICAL_KEYWORDS.get(lang, [])
    if not keywords:
        print(f"⚠️ No keyword list for {lang}, skipping.")
        return []

    selected = []
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                line = unicodedata.normalize("NFKC", line.strip())
                
                if len(line) < 30:
                    continue
                
                if not looks_medical(line):
                    continue
                
                line_lower = line.lower()
                keyword_count = sum(1 for keyword in keywords if keyword in line_lower)
                
                if keyword_count >= min_keywords:
                    selected.append(line)
                    
    except Exception as e:
        print(f"⚠️ Could not read {text_path}: {e}")
    
    print(f"🩺 {lang}: {len(selected)} medical lines found (min {min_keywords} keywords)")
    return selected


def to_meditron_jsonl(lines, out_path):
    """Save lines to JSONL format for Meditron pretraining."""
    with open(out_path, "w", encoding="utf-8") as f:
        for text in lines:
            f.write(json.dumps({"text": text, "modalities": []}, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(lines)} pretraining-formatted samples → {out_path}")


if __name__ == "__main__":
    merged_all = []
    MIN_KEYWORDS = 2
    
    for lang in LANGUAGES:
        print(f"\n=== 🌍 Processing {lang} ===")
        text_path = download_language(lang)
        if not text_path or not os.path.exists(text_path):
            continue
    
        medical_lines = filter_medical_lines(text_path, lang, min_keywords=MIN_KEYWORDS)
        if not medical_lines:
            continue
    
        out_jsonl = os.path.join(OUT_DIR, f"{lang}_medical_pretrain.jsonl")
        to_meditron_jsonl(medical_lines, out_jsonl)
        merged_all.extend([{"text": line, "modalities": []} for line in medical_lines])
    
    merged_path = os.path.join(OUT_DIR, "merged_medical_multilingual.jsonl")
    with open(merged_path, "w", encoding="utf-8") as f:
        for entry in merged_all:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n🌐 Merged multilingual Meditron corpus saved → {merged_path}")
    print(f"✅ Done with strict medical filtering (min {MIN_KEYWORDS} keywords per line).")
