import os
import csv
import itertools

csv_path = "src/multimeditron/experts/evaluation_pipeline/xray_data/Data_Entry_2017_randomized.csv"
images_dir = "src/multimeditron/experts/evaluation_pipeline/xray_data/images"

missing = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_name = row['Image Index']
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            missing.append(img_name)

print(f"Nombre d'images manquantes : {len(missing)}")
if missing:
    print(f"Exemples : {missing[:10]}")

print("Exemples d'images présentes dans le dossier images/:")
for img_file in itertools.islice(os.listdir(images_dir), 10):
    print(img_file)