"""
XLSum Medical Content Filter

Filters the GEM/xlsum multilingual dataset for medical content across all 44 languages.
Uses strict contextual filtering with language-specific blacklists and medical whitelists
to extract health-related articles while excluding politics, religion, sports, entertainment,
and other non-medical content.

Input: Locally extracted XLSum datasets (from .tar.bz2 archives)
Output: Filtered JSONL files per language in Meditron format

Usage:
    python convert_xlsum_to_pretraining.py
"""

import os
import re
import json
import unicodedata
from datasets import load_dataset
from tqdm import tqdm


BASE_DIR = "src/multimeditron/translation/datasets"
OUT_DIR = os.path.join(BASE_DIR, "formatted_datasets/healthcare_datasets/xlsum_medical_jsonl")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_KEYWORDS = 1
MIN_LENGTH = 100
FILTER_MODE = "strict"
SPLITS = ["train", "test"]


BLACKLIST_GENERAL = [
    "president", "minister", "government", "election", "parliament", "party", "prime minister",
    "senate", "congress", "politics", "protest", "strike", "army", "soldier", "war", "conflict",
    "rebel", "bomb", "attack", "violence", "police", "court", "judge", "trial",
    "religion", "church", "mosque", "synagogue", "temple", "bible", "quran", "allah", "jesus", "god",
    "imam", "pastor", "bishop", "christian", "muslim", "islamic", "faith", "prayer", "holy", "ramadan", "easter",
    "sport", "football", "soccer", "match", "goal", "cup", "olympic", "league", "cricket", "tennis", "rugby",
    "music", "film", "movie", "actor", "actress", "song", "concert", "festival", "award", "fashion",
    "culture", "royal", "king", "queen", "wedding", "family", "celebration", "funeral",
    "teacher", "school", "university", "education", "student", "exam",
    "technology", "economy", "finance", "trade", "market", "business", "company", "bank",
    "twitter", "facebook", "instagram", "bbc", "news", "video", "subscribe", "copyright", "breaking"
]

BLACKLIST_EXTRA = {
    "amharic": ["መንግስት", "ፕሬዝዳንት", "ፓርቲ", "ምርጫ", "ክርስትና", "ኢስላም", "መንፈስ", "ቤተክርስቲያን", "ሙዚቃ", "ፊልም", "ጨዋታ", "ስፖርት", "ባህል", "ትምህርት"],
    "arabic": ["حكومة", "وزير", "رئيس", "انتخابات", "برلمان", "كنيسة", "مسجد", "سياسة", "رياضة", "فيلم", "موسيقى", "كرة القدم", "ثقافة", "تعليم", "اقتصاد"],
    "azerbaijani": ["hökumət", "nazir", "prezident", "seçki", "siyasət", "məscid", "kilsə", "idman", "futbol", "musiqi", "film", "mədəniyyət", "təhsil"],
    "bengali": ["সরকার", "মন্ত্রী", "নির্বাচন", "রাজনীতি", "মসজিদ", "গির্জা", "খেলা", "ক্রিকেট", "ফুটবল", "ফিল্ম", "সংগীত", "সংস্কৃতি", "শিক্ষা"],
    "burmese": ["အစိုးရ", "ဝန်ကြီး", "သမ္မတ", "ရွေးကောက်ပွဲ", "နိုင်ငံရေး", "ဘုရားကျောင်း", "အစ္စလာမ်", "အားကစား", "ဘောလုံး", "ရုပ်ရှင်", "ဂီတ", "ယဉ်ကျေးမှု", "ပညာရေး"],
    "chinese_simplified": ["政府", "部长", "总统", "选举", "政治", "战争", "教堂", "清真寺", "足球", "体育", "电影", "音乐", "宗教", "文化", "教育", "经济"],
    "chinese_traditional": ["政府", "部長", "總統", "選舉", "政治", "戰爭", "教堂", "清真寺", "足球", "體育", "電影", "音樂", "宗教", "文化", "教育", "經濟"],
    "english": [],
    "french": ["président", "ministre", "élections", "gouvernement", "parlement", "religion", "église", "mosquée", "football", "film", "musique", "culture", "éducation", "économie"],
    "gujarati": ["સરકાર", "મંત્રી", "રાષ્ટ્રપતિ", "ચૂંટણી", "રાજકારણ", "મસ્જિદ", "ચર્ચ", "રમત", "ક્રિકેટ", "ફૂટબોલ", "સંગીત", "ફિલ્મ", "સંસ્કૃતિ", "શિક્ષણ"],
    "hausa": ["gwamnati", "minista", "siyasa", "jam'iyya", "zabe", "musulmi", "kirista", "wasa", "kwallon kafa", "waka", "fina-finai", "al'ada", "ilimi"],
    "hindi": ["सरकार", "मंत्री", "राष्ट्रपति", "चुनाव", "राजनीति", "मस्जिद", "गिरजाघर", "खेल", "क्रिकेट", "फुटबॉल", "फ़िल्म", "संगीत", "संस्कृति", "शिक्षा", "अर्थव्यवस्था"],
    "igbo": ["gọọmentị", "ndị ọchịchị", "ndị isi ala", "ntuli aka", "ndọrọndọrọ", "ụka", "alakụba", "egwuregwu", "bọọlụ", "ihe nkiri", "egwu", "omenala", "agụmakwụkwọ"],
    "indonesian": ["pemerintah", "menteri", "presiden", "pemilihan", "politik", "gereja", "masjid", "olahraga", "sepak bola", "musik", "film", "budaya", "pendidikan", "ekonomi"],
    "japanese": ["政府", "大臣", "首相", "選挙", "政治", "戦争", "教会", "モスク", "サッカー", "スポーツ", "映画", "音楽", "宗教", "文化", "教育", "経済"],
    "kirundi": ["guverinoma", "minisitiri", "perezida", "amatora", "politiki", "kiliziya", "umusigiti", "siporo", "umupira", "indirimbo", "filime", "umuco", "amashure"],
    "korean": ["정부", "장관", "대통령", "선거", "정치", "전쟁", "교회", "모스크", "축구", "스포츠", "영화", "음악", "종교", "문화", "교육", "경제"],
    "kyrgyz": ["өкмөт", "министр", "президент", "шайлоо", "саясат", "чиркөө", "мечит", "спорт", "футбол", "кино", "музыка", "дин", "маданият", "билим берүү"],
    "marathi": ["सरकार", "मंत्री", "राष्ट्रपती", "निवडणूक", "राजकारण", "मशीद", "चर्च", "खेळ", "क्रिकेट", "फुटबॉल", "चित्रपट", "संगीत", "संस्कृती", "शिक्षण"],
    "nepali": ["सरकार", "मन्त्री", "राष्ट्रपति", "निर्वाचन", "राजनीति", "मस्जिद", "गिर्जाघर", "खेल", "क्रिकेट", "फुटबल", "चलचित्र", "संगीत", "संस्कृति", "शिक्षा"],
    "oromo": ["mootummaa", "presidantii", "filannoo", "siyaasaa", "mana kiristaanaa", "masgiida", "taphataa", "ispoortii", "muuziqaa", "seenaa", "aadaa", "barnoota"],
    "pashto": ["حکومت", "وزیر", "ولسمشر", "ټاکنې", "سیاست", "کلیسا", "جومات", "سپورت", "فوټبال", "فلم", "موسیقی", "دین", "کلتور", "زده کړه"],
    "persian": ["دولت", "وزیر", "رئیس‌جمهور", "انتخابات", "سیاست", "کلیسا", "مسجد", "ورزش", "فوتبال", "فیلم", "موسیقی", "دین", "فرهنگ", "آموزش", "اقتصاد"],
    "pidgin": ["goment", "president", "minister", "election", "politics", "church", "mosque", "football", "sport", "music", "film", "movie", "game", "culture"],
    "portuguese": ["governo", "presidente", "ministro", "eleição", "eleições", "política", "igreja", "mesquita", "futebol", "esporte", "música", "filme", "religião", "cultura", "educação", "economia"],
    "punjabi": ["ਸਰਕਾਰ", "ਮੰਤਰੀ", "ਰਾਸ਼ਟਰਪਤੀ", "ਚੋਣ", "ਰਾਜਨੀਤੀ", "ਮਸਜਿਦ", "ਗਿਰਜਾਘਰ", "ਖੇਡ", "ਕ੍ਰਿਕਟ", "ਫੁੱਟਬਾਲ", "ਸੰਗੀਤ", "ਫ਼ਿਲਮ", "ਸੱਭਿਆਚਾਰ", "ਸਿੱਖਿਆ"],
    "russian": ["правительство", "министр", "президент", "выборы", "политика", "война", "церковь", "мечеть", "религия", "футбол", "спорт", "фильм", "музыка", "культура", "образование", "экономика"],
    "scottish_gaelic": ["riaghaltas", "ministear", "taghadh", "poilitigs", "eaglais", "mosg", "ball-coise", "spòrs", "film", "ceòl", "creideamh", "cultar", "foghlam"],
    "serbian_cyrillic": ["влада", "министар", "председник", "избори", "политика", "црква", "џамија", "религија", "фудбал", "спорт", "филм", "музика", "култура", "образовање"],
    "serbian_latin": ["vlada", "ministar", "predsednik", "izbori", "politika", "crkva", "džamija", "religija", "fudbal", "sport", "film", "muzika", "kultura", "obrazovanje"],
    "sinhala": ["රජය", "ඇමති", "ජනාධිපති", "මැතිවරණය", "දේශපාලනය", "දේවස්ථානය", "මුස්ලිම්", "ක්‍රිකට්", "පාපන්දු", "චිත්‍රපට", "සංගීතය", "ආගම", "සංස්කෘතිය", "අධ්‍යාපනය"],
    "somali": ["dawladda", "wasiir", "madaxwayne", "doorasho", "siyaasad", "kaniisad", "masaajid", "ciyaar", "kubadda cagta", "heeso", "film", "diin", "dhaqan", "waxbarasho"],
    "spanish": ["gobierno", "presidente", "ministro", "elecciones", "política", "iglesia", "mezquita", "fútbol", "deporte", "música", "película", "religión", "cultura", "educación", "economía"],
    "swahili": ["serikali", "waziri", "rais", "uchaguzi", "siasa", "kanisa", "msikiti", "mchezo", "mpira", "muziki", "filamu", "dini", "utamaduni", "elimu"],
    "tamil": ["அரசாங்கம்", "அமைச்சர்", "ஜனாதிபதி", "தேர்தல்", "அரசியல்", "தேவாலயம்", "மசூதி", "விளையாட்டு", "கிரிக்கெட்", "கால்பந்து", "திரைப்படம்", "இசை", "மதம்", "கலாச்சாரம்", "கல்வி"],
    "telugu": ["ప్రభుత్వం", "మంత్రి", "అధ్యక్షుడు", "ఎన్నికలు", "రాజకీయాలు", "చర్చి", "మసీదు", "క్రీడ", "క్రికెట్", "ఫుట్‌బాల్", "సినిమా", "సంగీతం", "మతం", "సంస్కృతి", "విద్య"],
    "thai": ["รัฐบาล", "รัฐมนตรี", "ประธานาธิบดี", "การเลือกตั้ง", "การเมือง", "โบสถ์", "มัสยิด", "กีฬา", "ฟุตบอล", "หนัง", "ดนตรี", "ศาสนา", "วัฒนธรรม", "การศึกษา"],
    "tigrinya": ["መንግስቲ", "ሚኒስተር", "ፕሬዝዳንት", "ምርጫ", "ፖለቲካ", "መንፈስ ቅዱስ", "ቤተ ክርስቲያን", "ስፖርት", "ኩዕሶ እግሪ", "ፊልም", "ሙዚቃ", "ሃይማኖት", "ባህሊ", "ትምህርቲ"],
    "turkish": ["hükümet", "bakan", "cumhurbaşkanı", "seçim", "siyaset", "kilise", "cami", "din", "futbol", "spor", "film", "müzik", "kültür", "eğitim", "ekonomi"],
    "ukrainian": ["уряд", "міністр", "президент", "вибори", "політика", "війна", "церква", "мечеть", "релігія", "футбол", "спорт", "фільм", "музика", "культура", "освіта", "економіка"],
    "urdu": ["حکومت", "وزیر", "صدر", "انتخابات", "سیاست", "گرجا", "مسجد", "کھیل", "کرکٹ", "فٹ بال", "فلم", "موسیقی", "مذہب", "ثقافت", "تعلیم"],
    "uzbek": ["hukumat", "vazir", "prezident", "saylov", "siyosat", "cherkov", "masjid", "sport", "futbol", "kino", "musiqa", "din", "madaniyat", "ta'lim"],
    "vietnamese": ["chính phủ", "bộ trưởng", "tổng thống", "bầu cử", "chính trị", "nhà thờ", "đền thờ", "thể thao", "bóng đá", "phim", "âm nhạc", "tôn giáo", "văn hóa", "giáo dục", "kinh tế"],
    "welsh": ["llywodraeth", "gweinidog", "arlywydd", "etholiad", "gwleidyddiaeth", "eglwys", "mosg", "pêl-droed", "chwaraeon", "ffilm", "cerddoriaeth", "crefydd", "diwylliant", "addysg"],
    "yoruba": ["ijoba", "minista", "aare", "idibo", "oselu", "ijo", "mọ́síláńmù", "ere", "bọọlu", "orin", "fiimu", "ẹsin", "aṣa", "ẹkọ"],
}

WHITELIST_GENERAL = [
    "hospital", "doctor", "nurse", "clinic", "health", "medicine", "medical",
    "surgery", "treatment", "infection", "disease", "virus", "fever", "covid",
    "coronavirus", "pandemic", "vaccine", "vaccination", "diagnosis", "cancer",
    "malaria", "aids", "hiv", "ebola", "cholera", "typhoid", "tuberculosis",
    "mental health", "pregnancy", "maternal", "childbirth", "diabetes", "blood",
    "pressure", "hypertension", "heart", "cardiac", "kidney", "renal", "liver", "hepatic",
    "lungs", "respiratory", "drug", "therapy", "antibiotic", "antiviral", "pharmacy",
    "public health", "epidemic", "outbreak", "midwife", "nursing", "oncology", "immunization",
    "patient", "illness", "symptom", "chronic", "acute", "pediatric", "geriatric"
]

WHITELIST_EXTRA = {
    "amharic": ["ሆስፒታል", "ሐኪም", "ዶክተር", "ነርስ", "ክሊኒክ", "ታካሚ", "መድሀኒት", "ሕክምና", "በሽታ", "ቫይረስ", "ኢንፌክሽን", "ኮቪድ", "ኮሮና", "ቫክሲን", "ክትባት", "ካንሰር", "ማለሪያ", "ኤይድስ", "ኢቦላ", "ወሊድ", "ሕፃን", "እናት", "ደም", "ግፊት", "ልብ", "ጉበት", "ኩላሊት", "አንጀት", "ሆድ", "የስኳር በሽታ"],
    "arabic": ["مستشفى", "طبيب", "ممرض", "ممرضة", "عيادة", "مريض", "صحة", "دواء", "علاج", "طبي", "جراحة", "مرض", "عدوى", "فيروس", "حمى", "كورونا", "كوفيد", "جائحة", "لقاح", "تطعيم", "تشخيص", "سرطان", "ملاريا", "إيدز", "إيبولا", "حمل", "أمومة", "ولادة", "قابلة", "رضيع", "سكري", "ضغط", "قلب", "كلية", "كبد", "رئة", "دم"],
    "english": [],
    "french": ["hôpital", "médecin", "docteur", "infirmier", "infirmière", "clinique", "patient", "santé", "médicament", "traitement", "médical", "chirurgie", "maladie", "infection", "virus", "fièvre", "coronavirus", "covid", "pandémie", "vaccin", "vaccination", "diagnostic", "cancer", "paludisme", "sida", "ebola", "grossesse", "maternité", "accouchement", "sage-femme", "nourrisson", "bébé", "diabète", "tension", "hypertension", "cœur", "rein", "foie", "poumon", "sang"],
    "spanish": ["hospital", "médico", "doctor", "enfermera", "enfermero", "clínica", "paciente", "salud", "medicina", "medicamento", "tratamiento", "médico", "cirugía", "enfermedad", "infección", "virus", "fiebre", "coronavirus", "corona", "covid", "pandemia", "vacuna", "vacunación", "diagnóstico", "cáncer", "malaria", "sida", "ébola", "embarazo", "embarazada", "maternidad", "parto", "comadrona", "bebé", "diabetes", "tensión", "presión arterial", "hipertensión", "corazón", "riñón", "hígado", "pulmón", "sangre"],
}


def normalize(text: str) -> str:
    """Normalize text using NFKC and clean whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def looks_non_medical(text: str, lang: str) -> bool:
    """Check if text contains blacklisted non-medical terms."""
    text_l = text.lower()
    for bad in BLACKLIST_GENERAL + BLACKLIST_EXTRA.get(lang, []):
        if bad.lower() in text_l:
            return True
    return False


def count_medical_terms(text: str, lang: str) -> int:
    """Count occurrences of medical keywords in text."""
    text_l = text.lower()
    terms = WHITELIST_GENERAL + WHITELIST_EXTRA.get(lang, [])
    return sum(1 for kw in terms if kw.lower() in text_l)


def is_medical(article: dict, lang: str) -> bool:
    """
    Determine if an article is medical based on filtering criteria.
    Applies length filter, blacklist check, and medical keyword count.
    """
    text = normalize(article.get("text", ""))
    summary = normalize(article.get("summary", ""))
    
    if len(text) < MIN_LENGTH:
        return False
    
    if looks_non_medical(text, lang) or looks_non_medical(summary, lang):
        return False
    
    hits_text = count_medical_terms(text, lang)
    
    if FILTER_MODE == "strict":
        return hits_text >= MIN_KEYWORDS
    else:
        return (hits_text >= 1) or (count_medical_terms(summary, lang) >= 1)


def save_jsonl(entries, path):
    """Save entries to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(entries)} articles → {path}")


def process_language(lang: str) -> bool:
    """Load XLSum locally, filter for medical content, and save."""
    print(f"\n=== 🌍 {lang} ===")
    local_dir = os.path.join(BASE_DIR, "raw/xlsum_local", lang)

    if not os.path.exists(local_dir):
        print(f"❌ Local dataset folder not found for {lang}: {local_dir}")
        return False

    try:
        ds = load_dataset(local_dir)
    except Exception as e:
        print(f"❌ Load failed for {lang}: {e}")
        return False

    for split in SPLITS:
        if split not in ds:
            print(f"  ⚠️ Split '{split}' not found for {lang}")
            continue

        kept = []
        total = len(ds[split])
        print(f"  Processing {split} split ({total} articles)...")
        for ex in tqdm(ds[split], desc=f"  Filtering {split}"):
            if is_medical(ex, lang):
                kept.append({
                    "text": normalize(ex.get("text", "")),
                    "modalities": []
                })

        out_path = os.path.join(OUT_DIR, lang, f"{split}.jsonl")
        save_jsonl(kept, out_path)
        ratio = (len(kept) / total * 100.0) if total else 0.0
        print(f"  ✅ {split}: kept {len(kept)}/{total} ({ratio:.2f}%)")

    return True


if __name__ == "__main__":
    print("🔍 Using locally extracted XLSum dataset")
    all_langs = sorted([
        "amharic", "arabic", "azerbaijani", "bengali", "burmese",
        "chinese_simplified", "chinese_traditional", "english", "french",
        "gujarati", "hausa", "hindi", "igbo", "indonesian", "japanese",
        "kirundi", "korean", "kyrgyz", "marathi", "nepali", "oromo",
        "pashto", "persian", "pidgin", "portuguese", "punjabi", "russian",
        "scottish_gaelic", "serbian_cyrillic", "serbian_latin", "sinhala",
        "somali", "spanish", "swahili", "tamil", "telugu", "thai",
        "tigrinya", "turkish", "ukrainian", "urdu", "uzbek", "vietnamese",
        "welsh", "yoruba"
    ])
    print(f"📊 Found {len(all_langs)} XLSum languages locally.")

    stats = {"processed": 0, "failed": 0}

    for lang in all_langs:
        ok = process_language(lang)
        if ok:
            stats["processed"] += 1
        else:
            stats["failed"] += 1

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Languages discovered: {len(all_langs)}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"\n✅ Completed strict medical filtering for XLSum languages.")
    print(f"📁 Output directory: {OUT_DIR}")
    print(f"📝 Files saved as: <language>/train.jsonl and <language>/test.jsonl")