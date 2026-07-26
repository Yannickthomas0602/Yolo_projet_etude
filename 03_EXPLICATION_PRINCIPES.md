# Principes expliqués : IA, classification et détection vectorielle

Ce document donne une explication claire et pédagogique des principes qui sous-tendent le système : comment fonctionne une IA de reconnaissance d'images, ce que sont les "embeddings" (vecteurs visuels) et comment on les utilize pour rechercher des images semblables.

## Ordre de lecture recommandé

Pour comprendre le project de la manière la plus pédagogique, voici l'ordre conseillé des documents (du premier au dernier) :

1. GUIDE_OISEAUX.md — Présentation utilisateur et bonnes pratiques pour les images
2. COMMENT_FONCTIONNE_IA_YOLO.md — Explication simplifiée du fonctionnement de YOLO pour ce project
3. EXPLICATION_PRINCIPES.md — Concepts théoriques (embeddings, similarité, FAISS)
4. MODE_CAMERA.md — Mode caméra : capture, thread, sauvegarde et UI
5. MODE_VECTERIEL.md — Mode vectoriel : CLIP + FAISS, indexation et recherche
6. SETUP_ENTRAINEMENT_OISEAUX.md — Instructions pour préparer et lancer l'entraînement
7. JOURNAL_PROJET_OISEAUX.md — Journal et historique des modifications (dernier à lire)

## 1) Apprendre à reconnaître : entraînement vs utilization

- Entraînement : on montre à l'ordinateur beaucoup d'exemples (photos étiquetées) et on le laisse ajuster des paramètres internes pour réduire ses erreurs. C'est la phase longue et coûteuse, réalisée une seule fois.
- Inférence (ou utilization) : une fois entraîné, le modèle peut rapidement analyzer une nouvelle image et proposer une prédiction. C'est ce qui se passe dans `analyse_oiseaux.py` quand on lui donne une photo.

Analogie : l'entraînement, c'est comme apprendre à un ornithologue en lui montrant des albums photo ; l'inférence, c'est l'ornithologue qui donne son avis quand on lui montre une nouvelle photo.

## 2) Comment une IA donne une prédiction (concept simplifié)

- Le modèle regarde l'image et calcule des nombres internes qui résument formes, couleurs, textures.
- À la fin, il produit un score pour chaque espèce possible (par ex. héron : 0.85, cormoran : 0.10, autre : 0.05).
- Le score le plus élevé devient la prédiction (top‑1). La "confiance" exprimée est ce score.

Important : ce score n'est pas une certitude absolute, mais une indication statistique basée sur ce que le modèle a appris.

## 3) Pourquoi on a besoin de seuils (BDD / INCERTITUDE / HORS_BDD)

- Un seuil élevé (ex. 0.60) signifie qu'on demande au modèle d'être assez sûr avant d'« accepter » la détection.
- Entre deux seuils, on marque l'image comme "douteuse" (INCERTITUDE) pour revue humaine.
- En dessous d'un seuil bas, on considère que l'image est probablement hors de la base de connaissances (HORS_BDD).

Ces règles permettent d'automatiser la gestion et d'éviter des faux positifs trop nombreux.

## 4) Embeddings et recherche par similarité (principe)

- Embedding : on convertit une image en une liste de nombres (un vecteur). Ce vecteur capture l'apparence globale de l'image.
- Normalization : on met ces vecteurs sur la même échelle (important pour comparer correctement).
- Measure de similarité : on compare deux vecteurs avec une opération mathématique (produit scalaire / cosinus). Plus le résultat est proche de 1, plus les images sont similaires.

Analogie : imagine des fiches où chaque fiche a des cases remplies (couleurs dominantes, formes...). Deux fiches proches signifient photos proches.

## 5) CLIP + FAISS : rôle de chacun

- CLIP : un modèle qui sait produire des embeddings visuels robustes (et aussi relier images et texte). On l'utilise pour transformer image → vecteur.
- FAISS : une bibliothèque optimisée pour retrouver rapidement les vecteurs les plus proches dans une grande base (index). Sans FAISS, la recherche serait trop lente sur des milliers d'images.

Workflow résumé : construire l'index (une opération hors ligne), puis pour chaque nouvelle image : calculer son vecteur (CLIP) et interroger FAISS pour obtenir les voisins les plus proches.

## 6) Pourquoi normaliser et utiliser produit intérieur (inner product)

- Si on normalize les vecteurs (longueur = 1), le produit intérieur devient équivalent au cosinus d'angle, une measure de similarité usuelle et efficace.
- FAISS peut utiliser ce produit intérieur pour rendre les recherches rapides et précises.

## 7) Comment combiner classification et similarité (idée pratique)

- Règle simple : si le classifieur est incertain (score faible) mais le voisin le plus proche a une similarité élevée, on peut considérer la prédiction comme "probable" et la passer en revue prioritairement.
- Cette combination n'est pas magique : elle améliore certains cas (ré-utilisation d'exemples) mais peut aussi renforcer des biais si la base contient des erreurs.

## 8) Limitations et précautions

- Domaine : un modèle entraîné sur un certain type de photos peut mal se comporter sur des images très différentes (luminosité, angle, espèces non présentes).
- Biais : si les examples d'entraînement sont déséquilibrés, les résultats seront biaisés.
- Similarité visuelle ne replace pas l'étiquette : deux images peuvent se ressembler sans représenter la même espèce (contexte, posture).

## 9) Points pratiques à retenir

- L'entraînement est séparé de l'utilisation. L'utilisateur ne « réentraîne » pas lors d'une capture.
- L'inférence est rapide, mais la rapidité dépend du matériel (GPU accélère beaucoup).
- FAISS rend la recherche visuelle scalable (utile si vous avez des milliers d'images).

---

Si tu veux, je peux transformer cette explication en une infographie simple (3 blocs : caméra → IA → recherche vectorielle) ou écrire une version très courte pour un public non spécialiste (1 page).

# Guide d'architecture — entraînement, tests, sons, index vectoriel

Ce document décrit une architecture recommandée pour organizer le project : entraînement, jeux de données, tests, gestion des fichiers audio, index vectoriel et intégration continue. Il vise la clarté, la reproductibilité et la maintenance.

## 1. Arborescence recommandée

- `yolov5/` : code source principal (déjà présent).
- `data/` : définitions de datasets (`*.yaml`) et scripts d'import.
- `dataset_oiseaux/` : images brutes organisées en `train/`, `val/`, `test/`.
- `enregistrements/` : sorties d'analyse triées par statut (`BDD/`, `INCERTITUDE/`, `HORS_BDD/`).
- `vectors/` : index FAISS et `mapping.json` (ex: `index.faiss`, `mapping.json`).
- `cri_predateur_ou_detresse/` : assets audio (sous-dossiers par espèce, fichiers `.mp3`).
- `runs/` : expériences d'entraînement et sorties (checkpoints, logs).
- `tests/` : tests unitaires et d'intégration exécutables par CI.
- `scripts/` : outils utilitaires (construction d'index, validation de dataset, export).

## 2. Séparation des responsabilités

- Code d'inférence (runtime) : tout ce qui est utilisé pour l'analyse en production (ex. `analyse_oiseaux.py`, `classify/predict.py`). Doit rester léger et robuste face aux dépendances optionnelles (CLIP/FAISS). Charger CLIP/FAISS uniquement si `vectors/index.faiss` existe.
- Code d'entraînement : scripts et configs (`train.py`, fichiers YAML de modèle). Ils peuvent utiliser plus de dépendances (GPU, apex, etc.).
- Outils hors ligne : indexation, prétraitement, augmentation, calibration des seuils — placés dans `scripts/`.
