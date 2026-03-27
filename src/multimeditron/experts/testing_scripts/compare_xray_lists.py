import os
import csv

csv_path = "src/multimeditron/experts/evaluation_pipeline/xray_data/Data_Entry_2017_randomized.csv"
images_dir = "src/multimeditron/experts/evaluation_pipeline/xray_data/images"

# Liste des images dans le dossier

# Fonction utilitaire pour normaliser les noms (casse et extension)
def normalize(name):
    return os.path.splitext(name.lower())[0]

# Liste des images sur disque (normalisées)
images_on_disk = set(normalize(f) for f in os.listdir(images_dir))

# Liste des images attendues par le CSV (normalisées)
images_in_csv = set()
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        images_in_csv.add(normalize(row['Image Index']))

# Images attendues mais absentes
missing = images_in_csv - images_on_disk
# Images présentes mais non référencées dans le CSV
extra = images_on_disk - images_in_csv

print(f"Images attendues par le CSV (normalisées) : {len(images_in_csv)}")
print(f"Images présentes sur disque (normalisées) : {len(images_on_disk)}")
print(f"Images manquantes (dans CSV mais pas sur disque) : {len(missing)}")
if missing:
    print("Exemples :", list(missing)[:10])
print(f"Images en trop (sur disque mais pas dans le CSV) : {len(extra)}")
if extra:
    print("Exemples :", list(extra)[:10])
