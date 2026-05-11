# Journal du projet de reconnaissance d'oiseaux

Ce document garde une trace simple et lisible de la démarche suivie sur le projet. Il résume la préparation du dataset, l'entraînement, la logique métier BDD / INCERTITUDE / HORS_BDD, puis la mise à jour de la documentation.

## 1. Objectif du projet

Le but est de disposer d'un prototype YOLOv5 capable de reconnaître 4 classes d'oiseaux et de servir de base à un système complet de reconnaissance avec décision métier et lecture audio.

## 2. Références utilisées

Les documents de référence du dépôt sont :

- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md)
- [README.md](README.md)
- [classify/tutorial.ipynb](classify/tutorial.ipynb)

## 3. Méthode de classement des images

### 3.1 Dossier source

Les images brutes étaient regroupées dans [Dataset](Dataset) avec 4 sous-dossiers d'origine :

- `balbuzard`
- `heron gris`
- `pixabay_cormorands`
- `pixabay_mouette_goélands`

### 3.2 Classes finales retenues

Le projet a été normalisé autour de 4 classes cibles :

- `balbuzard`
- `heron`
- `cormoran`
- `mouette_goeland`

La classe `mouette_goeland` regroupe volontairement les mouettes et les goélands, car ils étaient confondus dans le périmètre retenu.

### 3.3 Correspondance source vers classe

Le classement a suivi cette logique :

- `balbuzard` -> `balbuzard`
- `heron gris` -> `heron`
- `pixabay_cormorands` -> `cormoran`
- `pixabay_mouette_goélands` -> `mouette_goeland`

### 3.4 Répartition des données

Les images ont été réparties en :

- `train` : 70 %
- `validation` : 20 %
- `test` : 10 %

Le dataset classé a été généré dans [dataset_oiseaux](dataset_oiseaux).

### 3.5 Volume obtenu

Répartition approximative initiale :

- `balbuzard` : 400 images
- `heron gris` : 576 images
- `pixabay_cormorands` : 515 images
- `pixabay_mouette_goélands` : 559 images

Après répartition :

- `train` : 1434 images
- `validation` : 409 images
- `test` : 207 images

## 4. Actions réalisées sur le dépôt

### 4.1 Création du dataset classé

Le dossier [dataset_oiseaux](dataset_oiseaux) a été construit à partir du dossier brut [Dataset](Dataset).

### 4.2 Suppression du dossier brut

Le dossier brut [Dataset](Dataset) a ensuite été supprimé pour éviter de conserver les sources d'origine dans le dépôt.

### 4.3 Mise à jour de la documentation

Les guides ont été harmonisés pour refléter l'état réel du projet :

- 4 classes cibles,
- fusion de mouette et goeland,
- ajout de la logique BDD / INCERTITUDE / HORS_BDD,
- ajout d'un guide de lancement dédié.

### 4.4 Versionnement Git

Le dataset classé, les scripts utilitaires, les poids entraînés et la documentation ont été commités puis poussés sur GitHub.

## 5. Entraînement du modèle

### 5.1 Configuration retenue

L'entraînement a été lancé avec :

- modèle : `yolov5s-cls.pt`
- taille d'image : `224`
- batch size : `32`
- epochs : `30`
- device : `0` sur la RTX 4070

### 5.2 Point de blocage initial

La première tentative d'entraînement a échoué à cause d'un chemin de travail incorrect. Le script cherchait à être lancé depuis `C:\Users\yanni\Desktop\Yolo` au lieu du dossier [yolov5](.).

### 5.3 Correction GPU

Le venv a ensuite été mis à jour avec une version CUDA de PyTorch :

- `torch 2.6.0+cu124`
- `torchvision 0.21.0+cu124`
- `torchaudio 2.6.0+cu124`

La vérification a confirmé :

- `torch.cuda.is_available() = True`
- `torch.cuda.device_count() = 1`
- GPU détecté : `NVIDIA GeForce RTX 4070 Laptop GPU`

### 5.4 Entraînement réussi

L'entraînement de référence a été exécuté avec succès sur le dataset préparé.

- Résultats sauvegardés dans `runs/train-cls/exp4`
- Poids finaux : `runs/train-cls/exp4/weights/best.pt` et `runs/train-cls/exp4/weights/last.pt`

## 6. Logique métier ajoutée

Une logique de décision a été ajoutée dans [classify/predict.py](classify/predict.py) pour distinguer trois états :

- `BDD` : l'espèce est reconnue avec une confiance suffisante,
- `INCERTITUDE` : la confiance est intermédiaire,
- `HORS_BDD` : la confiance est trop faible pour valider la classe.

Cette logique est pilotée par deux seuils :

- seuil BDD : `0.60`
- seuil incertitude : `0.30`

## 7. Validation de l'inférence

Un test a été réalisé sur [pixabay_91570.jpg](dataset_oiseaux/train/cormoran/pixabay_91570.jpg).

Résultat observé :

- prédiction principale : `cormoran`
- confiance élevée
- statut métier validé selon le seuil choisi

Le comportement a aussi été vérifié avec des seuils plus stricts pour forcer les sorties `INCERTITUDE` puis `HORS_BDD`.

## 8. Mise à jour de la documentation

Les documents Markdown ont été remis en cohérence avec l'état actuel du projet.

### 8.1 Fichiers concernés

- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md)

### 8.2 Ce qui a été clarifié

- la démarche complète du projet,
- les chemins réels pour lancer l'IA,
- les commandes pour l'entraînement et l'inférence,
- la règle métier BDD / incertitude / hors BDD,
- le guide court de lancement.

## 9. État actuel du projet

À ce stade :

- le dataset brut n'existe plus dans le dépôt,
- le dataset classé est disponible dans [dataset_oiseaux](dataset_oiseaux),
- un modèle de référence est disponible dans `runs/train-cls/exp4`,
- la logique de décision est déjà intégrée,
- la documentation est regroupée autour d'un guide principal, d'un guide d'installation et d'un guide de lancement.

## 10. Prochaine étape possible

La prochaine étape naturelle est de brancher la décision métier sur la lecture audio et d'industrialiser le test final sur 1000 images variées.
