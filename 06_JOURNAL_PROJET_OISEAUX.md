# Journal du project — Reconnaissance d'oiseaux (YOLOv5)

Ce journal relate de manière chronologique les étapes, décisions et modifications importantes pour le project. Il vice à garder une trace utile pour les développeurs et les contributeurs.

---

## Contexte actuel

- Répertoire de travail : `yolov5/`
- Date : 20 mai 2026
- Objectif : prototype de reconnaissance d'oiseaux réunissant : classification YOLOv5, index vectoriel CLIP+FAISS, enregistrements structurés et lecture audio pour alertes.

---

## Chronologie synthétique des actions

### 1) Initialization et dataset

- Création du dépôt et premières versions du dataset `dataset_oiseaux/` (train/val/test) pour 4 classes.
- Nettoyage et normalization des labels (function `normalize_label` dans `analyse_oiseaux.py`).

### 2) Entraînement et expérimentations

- Plusieurs runs d'entraînement dans `runs/train-cls/` avec sauvegarde des meilleurs checkpoints (`best.pt`).
- Ajout de wrapper `train_wrapper.py` pour tester rapidement variations d'hyperparamètres.

### 3) Déploiement d'usage et analyze

- Écriture de `analyse_oiseaux.py` pour lancer `classify/predict.py`, parser la sortie et archiver les résultats.
- Ajout du mécanisme de classification BDD / INCERTITUDE / HORS_BDD avec seuils par défaut et post-traitement.

### 4) Index vectoriel et recherche visuelle

- Intégration de CLIP + FAISS via `vector_index.py` pour indexer `dataset_oiseaux` et `enregistrements`.
- Usage prévu : aider la décision métier quand le classifieur est incertain.

### 5) Lecture audio et UX

- Ajout des functions `find_audio_file`, `play_audio`, `play_bird_audio` pour jouer des cris d'oiseaux en mode image unique.
- Règle : pas de lecture audio en mode batch.

### 6) Réorganisation documentaire

- Numérotation et enrichissement des documents d'aide (`01_...` à `06_...`) pour clarifier l'ordre de lecture et les workflows.

---

## Décisions techniques et rationale

- **Seuils métier** : `BDD=0.60`, `INCERTITUDE=0.50` par défaut. Ces valeurs ont été choisies pour limiter les faux positifs tout en gardant une zone de revue humaine. À calibrer.
- **Vectoriel** : embeddings CLIP normalisés (L2) + `IndexFlatIP` pour inner product ≈ cosinus. Seuil initial recommandé `0.30`.
- **Audio** : multi-OS via `os.startfile` / `xdg-open` / `open`. Lecture contrôlée (cooldown ou changement de classe requis).
- **Azure** : intégration optionnelle (Blob + IoT Hub) activée uniquement si variables d'environnement renseignées.

---

## Fichiers et functions clés (référence rapide)

- `yolov5/analyse_oiseaux.py` : functions d'inférence, parsing, sauvegarde, audio, caméra, Azure.
- `yolov5/vector_index.py` : construction et interrogation d'index FAISS via CLIP.
- `yolov5/classify/predict.py` : script YOLOv5 appelé en sous-processus pour la classification.
- `cri_predateur_ou_detresse/` : assets audio organisés par espèce.

---

## Tests effectués et résultats

- Lecture audio : testée sur Windows via `os.startfile()` ; retours OK.
- Index vectoriel : pipeline de construction fonctionnelle localement (temps dépendant du GPU).
- Inference : parsing de la sortie YOLOv5 robuste pour l'usage prévu (post-traitement appliqué correctement).

---

## Problèmes ouverts et améliorations recommandées

- Centraliser les seuils dans un fichier de configuration (ex: `config.yaml`) plutôt que valeurs codées en dur.
- Ajouter tests unitaires pour `parse_prediction_line`, `find_audio_file`, `query_image`.
- Implémenter cooldown audio (timestamp) pour éviter lectures répétées en mode caméra.
- Ajouter un processus d'ajout incrémental dans `vector_index.py` pour éviter rebuild complete.

---

## Actions réalisées maintenant

- Réécriture et complétion de ce journal le 20/05/2026.

---

## Prochaines étapes (si tu veux que je les fasse)

1. Mettre à jour automatiquement tous les liens internes des fichiers Markdown vers leurs nouveaux noms.
2. Centraliser les paramètres (seuils, chemins) dans `config.yaml` et modifier les scripts pour les utiliser.
3. Ajouter tests unitaires et exécuter une passe de tests.
4. Commit & push des changements sur une branche `docs/renommer-docs`.

Dis‑moi la prochaine action à réaliser.
