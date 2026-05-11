# Guide complet du projet de reconnaissance d'oiseaux

Ce document rassemble les explications, les bonnes pratiques et les commandes utiles pour construire un système complet de reconnaissance d'oiseaux avec ce dépôt YOLOv5. Le prototype actuel est déjà organisé autour des 4 classes cibles, du dataset classé et d'une logique de décision BDD / incertitude / hors BDD.

Si tu veux lancer directement le modèle ou tester une image, commence par [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md). Si tu veux suivre la démarche complète, lis aussi [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md) et [JOURNAL_PROJET_OISEAUX.md](JOURNAL_PROJET_OISEAUX.md).

Le projet final doit couvrir trois étapes:

1. détecter l'oiseau dans une image ou un flux caméra,
2. identifier son espèce,
3. jouer automatiquement le son correspondant.

Contraintes fonctionnelles du prototype:

- le modèle doit gérer exactement 4 classes cibles,
- les sons sont déjà attribués à chaque espèce,
- les sons d'une même espèce doivent être joués aléatoirement,
- en test final, on évalue le comportement sur 1000 images d'oiseaux variés.

Le dépôt contient deux voies différentes:

- la détection d'objets avec YOLOv5, utilisée pour localiser l'oiseau dans l'image,
- la classification d'images, utile pour reconnaître l'espèce à partir d'une image déjà recadrée.

Le README du dépôt sépare clairement ces deux usages, avec une section détection et une section classification dans [README.md](README.md) et un notebook de classification dans [classify/tutorial.ipynb](classify/tutorial.ipynb).

## 1. Quel pipeline choisir

Pour un projet d'espèces d'oiseaux en temps réel, l'architecture la plus robuste est la suivante:

1. un détecteur localise l'oiseau dans l'image,
2. un classifieur identifie l'espèce sur le recadrage de l'oiseau,
3. une couche applicative joue le son associé à l'espèce détectée.

Cette séparation a plusieurs avantages:

- le détecteur apprend à trouver les oiseaux même dans des scènes complexes,
- le classifieur apprend à distinguer les espèces avec des images centrées sur l'animal,
- l'application finale reste plus simple à faire évoluer.

Si ton dataset contient uniquement une image par espèce, sans annotations de boîtes, tu peux commencer par la classification. Si tu veux détecter un oiseau dans une scène avec arrière-plan, il faudra annoter des boîtes de détection.

## 1.1 Démarche suivie dans ce dépôt

La démarche appliquée dans ce projet est simple et progressive :

1. classer les images du dossier brut dans 4 espèces finales,
2. normaliser les noms de classes pour éviter les problèmes de chemins,
3. séparer le dataset en train, validation et test,
4. entraîner un premier classifieur YOLOv5 sur GPU,
5. ajouter une logique métier pour distinguer `BDD`, `INCERTITUDE` et `HORS_BDD`,
6. valider l'inférence sur une image connue,
7. documenter la procédure pour pouvoir relancer le modèle sans ambiguïté.

L'objectif n'est pas seulement de produire un modèle, mais aussi de garder une trace claire de chaque étape pour qu'un futur lancement soit reproductible.

## 2. Gestion du dataset

Tu as déjà environ 650 images par espèce. C'est une bonne base pour démarrer, à condition de nettoyer le dataset avant l'entraînement.

Dans cette version, limite le périmètre à 4 classes exactement, mais garde en tête qu'à ce stade le dataset n'est pas encore complet. Le but est d'obtenir un prototype fiable et simple à valider, puis de compléter les classes manquantes.

### 2.1 Trier et nettoyer les images

Garde toujours une copie brute intacte du dataset original, puis travaille sur une copie propre.

À supprimer ou isoler:

- les images floues,
- les images trop sombres ou trop surexposées,
- les doublons exacts,
- les quasi-doublons provenant d'une rafale ou d'une vidéo,
- les images avec plusieurs oiseaux si tu fais de la classification pure,
- les images dont l'espèce est incertaine,
- les images où l'oiseau est trop petit, coupé ou masqué.

Bonnes pratiques utiles:

- vérifier manuellement un échantillon de chaque espèce,
- séparer les images provenant de la même source avant de faire le split train/validation/test,
- ne pas mélanger les images très proches d'une même scène entre les différents splits,
- supprimer les images mal étiquetées avant l'entraînement,
- garder des images variées en angle, lumière, fond et distance.

### 2.2 Organisation des dossiers

Si tu fais de la classification, la structure recommandée est celle-ci:

```txt
dataset/
├── train/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
├── validation/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
└── test/
    ├── moineau/
    ├── corbeau/
    ├── pigeon/
    ├── heron/
    ├── merle/
    └── aigrette/
```

Cette structure correspond au mode classification de YOLOv5, comme indiqué dans [README.md](README.md#L364) et dans [data/ImageNet.yaml](data/ImageNet.yaml).

Si tu fais de la détection, la structure doit être différente:

```txt
dataset/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
└── labels/
    ├── train/
    ├── validation/
    └── test/
```

Dans ce cas, chaque image possède un fichier texte d'annotation au format YOLO.

### 2.3 Répartition recommandée

Un bon point de départ est:

- 70 % train,
- 20 % validation,
- 10 % test.

Si tes images proviennent de vidéos ou de rafales, essaie de regrouper les images très proches dans le même split. Le but est d'éviter que le modèle voie presque la même scène pendant l'entraînement et pendant la validation.

## 3. Préparation du dataset de classification

Le mode classification est le plus simple pour démarrer si tu veux reconnaître l'espèce à partir d'un oiseau déjà visible dans l'image.

### 3.1 Structure finale attendue

Exemple:

```txt
dataset_oiseaux/
├── train/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
├── validation/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
└── test/
    ├── moineau/
    ├── corbeau/
    ├── pigeon/
    ├── heron/
    ├── merle/
    └── aigrette/
```

Chaque sous-dossier correspond à une classe.

### 3.2 Script de tri automatique simple

Le script ci-dessous prend un dossier brut organisé par espèce, filtre les images très floues, puis génère train, validation et test.

```python
from pathlib import Path
import random
import shutil

import cv2

RAW_DIR = Path(r"C:\Users\yanni\Desktop\Yolo\dataset_raw")
OUT_DIR = Path(r"C:\Users\yanni\Desktop\Yolo\dataset_oiseaux")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
BLUR_THRESHOLD = 100.0
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_blurry(image_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD


def list_images(folder: Path) -> list[Path]:
    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_split(files: list[Path], split_name: str, class_name: str) -> None:
    target_dir = OUT_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(RAW_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        images = list_images(class_dir)
        clean_images = [img for img in images if not is_blurry(img)]

        random.shuffle(clean_images)

        total = len(clean_images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_files = clean_images[:train_count]
        val_files = clean_images[train_count:train_count + val_count]
        test_files = clean_images[train_count + val_count:]

        copy_split(train_files, "train", class_dir.name)
        copy_split(val_files, "validation", class_dir.name)
        copy_split(test_files, "test", class_dir.name)

        print(
            f"{class_dir.name}: {total} images propres -> "
            f"{len(train_files)} train, {len(val_files)} validation, {len(test_files)} test"
        )


if __name__ == "__main__":
    main()
```

Ce script ne remplace pas le contrôle manuel. Il sert surtout à te faire gagner du temps sur la première version du dataset.

## 4. Préparation du dataset de détection

Si tu veux détecter un oiseau dans une image ou dans un flux caméra, il faut annoter des boîtes englobantes.

### 4.1 Format des annotations

Chaque image doit avoir un fichier texte portant le même nom que l'image.

Exemple:

```txt
images/train/oiseau_001.jpg
labels/train/oiseau_001.txt
```

Le fichier texte contient des lignes au format YOLO:

```txt
classe centre_x centre_y largeur hauteur
```

Les coordonnées sont normalisées entre 0 et 1.

Exemple:

```txt
0 0.512 0.438 0.220 0.315
```

### 4.2 Fichier YAML de détection

Exemple minimal:

```yaml
path: C:/Users/yanni/Desktop/Yolo/datasets/birds
train: images/train
val: images/validation
test: images/test

names:
  0: moineau
  1: corbeau
    2: pigeon
    3: heron
    4: merle
    5: aigrette
```

Le fichier doit pointer vers les dossiers d'images et lister les classes dans le bon ordre.

## 5. Entraînement avec YOLOv5

Le dépôt indique dans [README.md](README.md) les commandes de base pour l'entraînement, la validation et la détection.

### 5.1 Activer l'environnement virtuel

Dans ce workspace, le venv se trouve dans [yolov5/.venv](.venv).

```powershell
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
.\.venv\Scripts\Activate.ps1
```

### 5.2 Installer les dépendances

```powershell
python -m pip install -r requirements.txt
```

### 5.3 Entraîner un modèle de classification

La section classification du README explique que YOLOv5 supporte l'entraînement de classifieurs depuis [README.md](README.md#L364).

Exemple de lancement:

```powershell
python classify/train.py --model yolov5s-cls.pt --data dataset_oiseaux --epochs 20 --img 224 --batch 32
```

Adapte `dataset_oiseaux` au nom réel de ton dossier si tu le places ailleurs.

### 5.4 Entraîner un modèle de détection

Si tu as des annotations de boîtes, tu peux lancer:

```powershell
python train.py --img 640 --batch 16 --epochs 100 --data birds.yaml --weights yolov5s.pt
```

La section détection du README montre la même logique avec le fichier [data/coco.yaml](data/coco.yaml#L1).

### 5.5 Valider le modèle

```powershell
python val.py --data birds.yaml --weights runs/train/exp/weights/best.pt
```

Pour la classification, utilise:

```powershell
python classify/val.py --data dataset_oiseaux --weights runs/train-cls/exp/weights/best.pt --img 224
```

## 6. Architecture propre du projet

Je te recommande cette organisation:

```txt
project/
├── dataset_raw/
├── dataset_oiseaux/
├── dataset_detection/
├── models/
├── sounds/
│   ├── heron/
│   │   ├── son_heron1.mp3
│   │   ├── son_heron2.mp3
│   │   └── son_heron3.mp3
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── merle/
│   └── aigrette/
├── scripts/
├── config/
├── runs/
└── app/
```

Rôle de chaque dossier:

- `dataset_raw`: images brutes non modifiées,
- `dataset_oiseaux`: dataset de classification nettoyé,
- `dataset_detection`: dataset annoté pour la détection,
- `models`: poids entraînés et exports,
- `sounds`: fichiers audio des espèces rangés par dossier d'espèce,
- `config`: fichier de correspondance entre espèce et sons,
- `scripts`: préparation des données et automatisations,
- `app`: intégration caméra, détection et lecture audio.

Pour ton cas, un stockage simple et local est le plus judicieux. Comme les sons sont déjà attribués à chaque espèce, il n'est pas utile de mettre en place une grosse base de données. La solution la plus pratique pour un prototype est:

- un dossier par espèce dans `sounds/`,
- un fichier de configuration simple en JSON ou YAML qui associe chaque espèce à la liste de ses sons,
- une logique applicative qui choisit un son aléatoire dans la liste de l'espèce détectée.

Exemple de configuration en YAML:

```yaml
heron:
    - sounds/heron/son_heron1.mp3
    - sounds/heron/son_heron2.mp3
    - sounds/heron/son_heron3.mp3
moineau:
    - sounds/moineau/son_moineau1.mp3
    - sounds/moineau/son_moineau2.mp3
pigeon:
    - sounds/pigeon/son_pigeon1.mp3
```

Cette approche est simple à maintenir, facile à modifier à la main et suffisante pour un prototype. Elle évite la complexité d'une base de données scalable qui ne serait pas utile à ce stade.

## 7. Pipeline temps réel

Le système final peut fonctionner ainsi:

1. capture d'une image ou d'une frame caméra,
2. détection de l'oiseau,
3. recadrage de la zone détectée,
4. classification de l'espèce,
5. récupération dans le fichier de configuration de la liste des sons associés à l'espèce,
6. sélection aléatoire d'un son parmi les fichiers disponibles pour cette espèce,
7. lecture du son si la confiance dépasse un seuil,
8. stockage éventuel du résultat dans un journal.

Règles métier de lecture audio à appliquer:

1. si l'oiseau détecté appartient à l'une des 4 classes et que la confiance est suffisante, jouer un son choisi aléatoirement parmi les sons de cette classe,
2. si l'oiseau détecté n'appartient pas aux 4 classes cibles, ne jouer aucun son,
3. si le modèle n'est pas sûr (zone d'incertitude), jouer un son de base unique,
4. appliquer un court délai anti-répétition pour éviter un déclenchement audio en boucle.

Règle de décision simple pour la classification:

- si la confiance top-1 est supérieure ou égale au seuil BDD, l'oiseau est traité comme une espèce connue du périmètre,
- si la confiance est comprise entre le seuil d'incertitude et le seuil BDD, le système retourne une incertitude,
- si la confiance est inférieure au seuil d'incertitude, le système considère l'oiseau comme hors BDD.

Les deux seuils sont à ajuster selon les essais; pour démarrer, une base pratique est `BDD >= 0.60` et `incertitude >= 0.30`, soit 60 % pour BDD et 30 % pour l'incertitude.

Bonnes pratiques temps réel:

- ajouter un seuil de confiance avant de jouer un son,
- éviter de rejouer le son à chaque frame,
- garder une mémoire courte pour ne pas répéter le même son en boucle,
- mesurer la latence entre la capture et la prédiction,
- utiliser un modèle léger si l'exécution doit être fluide en webcam.

Pour un prototype, je recommande cette logique de stockage:

- stocker les fichiers audio localement dans le dépôt,
- garder une seule source de vérité dans un fichier YAML ou JSON,
- utiliser des noms de dossiers et de fichiers simples,
- ne pas introduire de base de données tant que le périmètre reste expérimental.

Le caractère aléatoire doit rester strictement par espèce: on ne tire jamais un son d'une autre espèce.

Exemple de règle d'exécution:

1. l'espèce détectée vaut `heron`,
2. l'application lit la liste des sons pour `heron` dans la configuration,
3. elle choisit aléatoirement `son_heron2.mp3`,
4. elle le joue une seule fois,
5. elle attend un délai avant de pouvoir rejouer un son identique.

## 8. Liste de contrôle avant entraînement

Avant de lancer l'entraînement, vérifie:

- chaque espèce a un nombre d'images à peu près équilibré,
- les images floues ou cassées ont été retirées,
- les doublons ont été réduits,
- le split train/validation/test est cohérent,
- les noms de classes sont identiques partout,
- le format des dossiers correspond au type de tâche choisi,
- le venv est actif et les dépendances sont installées.

## 9. Test final du prototype (1000 images)

Objectif: estimer le taux d'erreur dans une situation réaliste.

Jeu de test final recommandé:

- 1000 images d'oiseaux variés,
- mélange d'images appartenant aux 4 classes cibles et d'images d'autres espèces,
- images non vues pendant l'entraînement.

Mesures à suivre:

- taux de bonne identification pour les 4 classes,
- taux de faux positifs sur les espèces hors périmètre,
- taux de déclenchement du son de base (incertitude),
- taux de cas où aucun son est joué correctement pour les classes hors des 4 classes.

Comportement attendu:

1. oiseau des 4 classes et confiance suffisante: son aléatoire de la classe,
2. oiseau hors des 4 classes: aucun son,
3. confiance insuffisante: son de base.

## 10. Commandes utiles

Activer le venv:

```powershell
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances:

```powershell
python -m pip install -r requirements.txt
```

Vérifier l'interpréteur actif:

```powershell
python -c "import sys; print(sys.executable)"
```

Lancer une classification:

```powershell
python classify/train.py --model yolov5s-cls.pt --data dataset_oiseaux --epochs 20 --img 224 --batch 32
```

Lancer une détection:

```powershell
python train.py --img 640 --batch 16 --epochs 100 --data birds.yaml --weights yolov5s.pt
```

## 11. Recommandation pratique

Si tu débutes, fais d'abord une version classification simple avec tes images déjà triées par espèce. Ensuite, si tu veux localiser précisément les oiseaux dans des scènes réelles, ajoute un second dataset de détection avec annotations.

Cette approche t'évite de te battre dès le départ avec l'annotation de boîtes, tout en te donnant une base exploitable pour le système complet.

Comme tu précises que le projet est un prototype, garde le stockage des sons volontairement simple: dossiers locaux, fichiers MP3 rangés par espèce, et mapping YAML ou JSON. C'est le meilleur compromis entre facilité d'usage et lisibilité du code.
