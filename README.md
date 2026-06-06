# Projet Oiseaux — YOLOv5

Description
- Ce répertoire contient une adaptation locale de YOLOv5 dédiée à l'analyse et la classification d'oiseaux, avec les scripts d'entraînement, d'inférence et la documentation spécifique au projet. Ce projet a un but éducatif et pédagogique uniquement, et n'a pas vocation à être commercialisé ni produit à grande échelle.

Contexte et documentation essentielle
- [01_GUIDE_OISEAUX.md](01_GUIDE_OISEAUX.md) — Guide d'utilisation spécifique au projet oiseaux
- [02_COMMENT_FONCTIONNE_IA_YOLOV5.md](02_COMMENT_FONCTIONNE_IA_YOLOV5.md) — Explication technique du fonctionnement
- [03_EXPLICATION_PRINCIPES.md](03_EXPLICATION_PRINCIPES.md)
- [04_MODE_CAMERA_OISEAUX.md](04_MODE_CAMERA_OISEAUX.md)
- [05_MODE_VECTORIEL.md](05_MODE_VECTORIEL.md)
- [06_JOURNAL_PROJET_OISEAUX.md](06_JOURNAL_PROJET_OISEAUX.md)
- [07_EXPLICATION_ANALYSE_VECTORIELLE.md](07_EXPLICATION_ANALYSE_VECTORIELLE.md)
- [08_ANALYSE_VECTORIELLE_VULGARISEE.md](08_ANALYSE_VECTORIELLE_VULGARISEE.md)

**Livrable**
- Ce projet s'accompagne d'un livrable décrivant le déroulé complet du travail réalisé :

- 1. Recherche et cadrage à partir du sujet principal « IA et Agriculture ». 
- 2. Veille ciblée sur la pisciculture et les problématiques liées aux attaques d'oiseaux sur les bassins. 
- 3. Conception d'une solution technique pour réduire les attaques (stratégies de détection/alerte et mesures de protection). 
- 4. Début du développement et validation initiale : entraînement d'une IA capable d'analyser et classifier les espèces d'oiseaux piscivores.

- Le livrable contient le rapport de recherche, les choix techniques, les jeux de données utilisés et les premiers résultats d'entraînement. Il accompagne le projet et peut être livré dans un dossier `livrable/` ou via une release GitHub selon votre préférence.

Voir le livrable : [Livrable_CIAeau_Oiseaux.pdf](livrable/Livrable_CIAeau_Oiseaux.pdf)

Fichiers et dossiers importants (dans ce répertoire)
- `data/` et `dataset_oiseaux/` : jeux de données (train/validation/test)
- `enregistrements/` : enregistrements audio et dossiers d'analyse
- `models/` : définitions de modèles et poids
- `runs/` : sorties d'entraînement et résultats
- `classify/` : scripts de classification
- `scripts/`, `utils/` : utilitaires et scripts divers

Exemples rapides (depuis ce répertoire)
- Créer et activer l'environnement virtuel (Windows PowerShell) :

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned)
& .\.venv\Scripts\Activate.ps1
```

- Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
```

- Lancer une détection sur une image :

```powershell
python detect.py --source path/to/image.jpg --weights yolov5s-cls.pt
```

Yolov5
- Voir [README_Yolov5.md](README_Yolov5.md) pour plus d'informations sur Yolov5.


