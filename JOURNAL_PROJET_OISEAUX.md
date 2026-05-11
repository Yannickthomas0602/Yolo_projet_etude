# Journal du projet de reconnaissance d'oiseaux

Ce document suit, étape par étape, ce qui a été fait sur le projet. Il sert de trace claire pour comprendre les choix de classement des images, la préparation du dataset et les actions techniques réalisées.

## 1. Objectif du projet

Le but est de préparer un modèle YOLOv5 capable de reconnaître les oiseaux présents dans le dataset du projet et de servir de base à l'entraînement de l'IA.

## 2. Références utilisées

Les consignes et l'organisation du projet ont été construites à partir de :
- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- [README.md](README.md)
- [classify/tutorial.ipynb](classify/tutorial.ipynb)

## 3. Méthode de classement des images

### 3.1 Dossier source

Les images étaient initialement regroupées dans le dossier [Dataset](Dataset), avec 4 sous-dossiers sources :
- `balbuzard`
- `heron gris`
- `pixabay_cormorands`
- `pixabay_mouette_goélands`

### 3.2 Classes finales retenues

Après analyse du contenu et des besoins du projet, les images ont été classées dans 4 classes cibles :
- `balbuzard`
- `heron`
- `cormoran`
- `mouette_goeland`

La classe `mouette_goeland` regroupe volontairement les mouettes et les goélands, car ils ont été confondus dans le périmètre retenu.

### 3.3 Correspondance source → classe

Le classement effectué a suivi cette logique :
- `balbuzard` → `balbuzard`
- `heron gris` → `heron`
- `pixabay_cormorands` → `cormoran`
- `pixabay_mouette_goélands` → `mouette_goeland`

### 3.4 Répartition des données

Les images ont été réparties automatiquement en :
- `train` : 70 %
- `validation` : 20 %
- `test` : 10 %

La structure générée est :

```txt
dataset_oiseaux/
├── train/
├── validation/
└── test/
```

Chaque split contient les 4 classes ci-dessus.

### 3.5 Volume obtenu

Répartition approximative obtenue lors du tri :
- `balbuzard` : 400 images au total
- `heron gris` : 576 images au total
- `pixabay_cormorands` : 515 images au total
- `pixabay_mouette_goélands` : 559 images au total

Après répartition, le dataset classé contenait :
- `train` : 1434 images
- `validation` : 409 images
- `test` : 207 images

## 4. Actions réalisées sur le dépôt

### 4.1 Création du dataset classé

Le dossier [dataset_oiseaux](dataset_oiseaux) a été créé à partir du dossier brut [Dataset](Dataset).

### 4.2 Suppression du dossier brut

Après vérification que la copie classée était bien complète, le dossier [Dataset](Dataset) a été supprimé pour éviter de conserver les sources brutes dans le dépôt.

### 4.3 Mise à jour de la documentation

Les guides ont été harmonisés pour refléter la réalité du dataset :
- 4 classes au lieu de 6 ou 5
- fusion de mouette et goeland en une seule classe
- cohérence entre le guide principal et le guide d'installation

### 4.4 Versionnement Git

Le dataset classé a été ajouté au dépôt Git, puis commit et push vers GitHub.

## 5. État actuel du projet

À ce stade :
- le dataset brut n'existe plus dans le dépôt
- le dataset classé est disponible dans [dataset_oiseaux](dataset_oiseaux)
- les guides décrivent les 4 classes réelles du projet
- le commit a été poussé sur GitHub

## 6. Convention à suivre pour la suite

Quand une nouvelle demande arrive, les actions doivent être expliquées et enregistrées ici si elles modifient le projet.

Les points importants à conserver :
- expliquer la logique du classement si une nouvelle classe apparaît
- documenter les splits train/validation/test
- noter toute modification de structure ou de nom de classe
- garder une trace des commits importants

## 7. Prochaine étape possible

La prochaine étape naturelle est l'entraînement du modèle sur [dataset_oiseaux](dataset_oiseaux), puis la validation des résultats.
