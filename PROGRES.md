# Progress

Write your achievements here:

- Made training_new_pipeline work
- Made ultrasound new benchmark work 

---

# Next steps

List upcoming tasks here:

- Add the 2 new datasets (load them in stream not fully)
- Fix X-ray benchmark
- Clean messy paths !!
- Test more systematically 
- Check the data for the benchmark
- Check for the paths and the files overall for the benchmark
- Do the configs and config maker systematically
- add other benchmark
- fix and standardize the config files


# Accomplishment week 14.03-20.03
- Made xray_eval work


# Different benchmark

- *general* : base_clip_evaluation.py base_sim_benchmark.py,hard_negatives_evaluation.py, hard_benchmark_skin_tone_stratified.py,, display_most_sim.py, check_negative_overlap.py,load_from_clip.py
- *skin diseases* : folder disease_classification_pipeline
- *ophtalmology* : ophthamology_benchmark_raw.py
- *ultrasound* : ultrasound_new_benchmark.py
- *xray* : xray_eval.py
- *MRI* : mri_benchmark_raw.py
- *CT scan* : ct_benchmark_raw.py 



Advice : 

1. Économiser du compute et tester la pipeline
Utilise des petits échantillons : crée des versions réduites de chaque dataset (quelques dizaines/centaines d’exemples) pour tester la pipeline de bout en bout rapidement.
Mode dry-run / debug : si la pipeline le permet, active un mode “debug” ou “dry-run” qui ne fait qu’un passage ou saute l’entraînement, mais vérifie le chargement des données, la création des modèles, etc.
Unit tests sur les modules critiques : écris des tests unitaires pour les fonctions de chargement, de préprocessing, et de batching.
Test sur un batch unique : lance la pipeline pour un seul batch (ou une seule epoch) pour vérifier que tout passe sans erreur.
Mock datasets : crée des datasets factices avec la même structure que les vrais pour tester le code sans dépendre des données réelles.
2. Pour les datasets problématiques (ex : XR-glob)
Streaming : si le dataset est trop gros, utilise le streaming (ex : avec HuggingFace datasets ou DataLoader en mode streaming) pour ne pas tout charger en RAM.
Logs détaillés : ajoute des logs sur les étapes de chargement pour repérer où ça bloque.
Vérifie les chemins : assure-toi que les chemins des datasets sont corrects et que les fichiers existent.
3. Travailler de manière systématique
Checklist de tests : fais une liste des datasets et des benchmarks à tester, coche-les au fur et à mesure.
Automatisation : écris des scripts pour lancer les tests sur tous les datasets en mode rapide (échantillon réduit, 1 epoch).
Versionne tes configs : garde une trace des configurations qui marchent pour chaque dataset.
Documente tes progrès : note ce qui fonctionne, ce qui plante, et les solutions trouvées (dans PROGRES.md par exemple).
4. Conseils généraux
Valide chaque étape séparément : teste d’abord le chargement, puis le préprocessing, puis l’entraînement, etc.
Utilise des assertions : vérifie la forme et le contenu des batchs à chaque étape.
Sauvegarde régulière : commit souvent, surtout après avoir résolu un bug ou validé une étape.
5. Pour le problème du dataset local
Solution temporaire : crée un petit dataset local “dummy” pour passer l’étape de chargement.
Refactorise le code : si possible, modifie la pipeline pour accepter des datasets distants ou en streaming.
Résumé systématique :

Crée des mini-datasets pour chaque benchmark.
Lance la pipeline sur un batch/epoch pour chaque dataset.
Ajoute des logs et assertions à chaque étape.
Automatiser les tests rapides sur tous les datasets.
Documente chaque succès/échec et solution.