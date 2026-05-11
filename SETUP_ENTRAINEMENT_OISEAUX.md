# Guide Complet : Mise en Place de l'Entraînement YOLOv5 pour la Reconnaissance d'Oiseaux

Ce guide vous explique **étape par étape** comment mettre en place l'entraînement d'une Intelligence Artificielle pour reconnaître 6 espèces d'oiseaux à partir du projet YOLOv5. **Pour l'instant, le dataset ne contient pas encore les 6 espèces complètes** : ce document décrit donc la cible finale et la manière de préparer le projet pour l'atteindre.

---

## Table des Matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Préparation de l'environnement](#2-préparation-de-lenvironnement)
3. [Gestion et nettoyage du dataset](#3-gestion-et-nettoyage-du-dataset)
4. [Organisation des dossiers](#4-organisation-des-dossiers)
5. [Entraînement du modèle](#5-entraînement-du-modèle)
6. [Validation et test](#6-validation-et-test)
7. [Pipeline temps réel](#7-pipeline-temps-réel)
8. [Checklist avant lancement](#8-checklist-avant-lancement)

---

## 1. Vue d'ensemble du projet

### 1.1 Objectif global

Créer un système complet de reconnaissance d'oiseaux qui :
1. **Détecte** l'oiseau dans une image ou un flux vidéo
2. **Identifie** son espèce parmi 6 espèces cibles
3. **Joue** automatiquement un son correspondant

### 1.2 Les 6 espèces cibles

Le prototype vise à gérer exactement ces 6 espèces, mais **à ce stade nous ne les avons pas encore toutes dans le dataset** :
- **Moineau** (*Passer domesticus*)
- **Corbeau** (*Corvus*)
- **Pigeon** (*Columba*)
- **Héron** (*Ardea*)
- **Merle** (*Turdus merula*)
- **Aigrette** (*Egretta*)

L'objectif du guide est donc de préparer le pipeline de façon progressive, en partant des espèces déjà disponibles puis en complétant les classes manquantes.

### 1.3 Approche recommandée : Classification

Pour ce projet, nous utilisons une approche en **2 étapes** :

1. **Détecteur** : localise l'oiseau dans l'image (boîte englobante)
2. **Classifieur** : identifie l'espèce sur le recadrage de l'oiseau

Cette approche offre plusieurs avantages :
- Le détecteur apprend à localiser les oiseaux dans des scènes complexes
- Le classifieur se concentre sur la distinction des espèces avec des images centrées
- Le système est plus simple à évolver

### 1.4 Données disponibles

Vous disposez d'environ **650 images par espèce**, ce qui est une bonne base pour démarrer.

---

## 2. Préparation de l'environnement

### 2.1 Localisation du workspace

Le dépôt YOLOv5 se trouve ici :
```
C:\Users\yanni\Desktop\Yolo\yolov5\
```

### 2.2 Activation de l'environnement virtuel

Ouvrez PowerShell et naviguez dans le dépôt :

```powershell
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
```

Activez l'environnement virtuel :

```powershell
.\.venv\Scripts\Activate.ps1
```

Vous devriez voir `(.venv)` apparaître au début de votre terminal.

### 2.3 Installation des dépendances

Une fois le venv activé, installez les packages requis :

```powershell
python -m pip install -r requirements.txt
```

Cela peut prendre quelques minutes.

### 2.4 Vérification de l'installation

Vérifiez que l'installation est correcte :

```powershell
python -c "import torch; print(torch.__version__)"
```

Vous devriez voir un numéro de version PyTorch (ex: `2.0.0`).

---

## 3. Gestion et nettoyage du dataset

### 3.1 Pourquoi nettoyer ?

Avant l'entraînement, vous devez nettoyer votre dataset pour éliminer les images problématiques. Cela améliore la qualité de l'apprentissage du modèle.

### 3.2 Images à supprimer ou isoler

Voici les types d'images à écarting :

| Type | Raison |
|------|--------|
| Images floues | Confondent le modèle |
| Images trop sombres / surexposées | Perte d'information |
| Doublons exacts | Biaissent l'entraînement |
| Quasi-doublons d'une rafale | Favorisent les données similaires |
| Plusieurs oiseaux (classification) | Compliquent la classification |
| Espèce incertaine | Introduction de bruit dans les labels |
| Oiseau trop petit / coupé / masqué | Difficiles à classer |

### 3.3 Bonnes pratiques de nettoyage

- **Vérifiez manuellement** un échantillon de 20-30 images par espèce
- **Séparez** les images provenant de la même source avant le split train/validation/test
- **Isolez** les images similaires d'une même scène dans le même ensemble (train ou validation, pas des deux)
- **Supprimez** les labels incorrects
- **Conservez** une variété : angles différents, conditions d'éclairage, arrière-plans, distances

### 3.4 Sauvegarde de la version originale

**Toujours** garder une copie de vos données brutes :

```
C:\Users\yanni\Desktop\Yolo\dataset_raw\          (original, jamais modifié)
C:\Users\yanni\Desktop\Yolo\dataset_oiseaux\      (copie nettoyée pour l'entraînement)
```

### 3.5 Script Python pour trier et nettoyer automatiquement

Créez un fichier `scripts/clean_dataset.py` avec ce contenu :

```python
from pathlib import Path
import random
import shutil
import cv2

# Configuration
RAW_DIR = Path(r"C:\Users\yanni\Desktop\Yolo\dataset_raw")
OUT_DIR = Path(r"C:\Users\yanni\Desktop\Yolo\dataset_oiseaux")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
BLUR_THRESHOLD = 100.0
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_blurry(image_path: Path) -> bool:
    """Détecte si une image est trop floue."""
    image = cv2.imread(str(image_path))
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD


def list_images(folder: Path) -> list[Path]:
    """Liste toutes les images dans un dossier."""
    return [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_split(files: list[Path], split_name: str, class_name: str) -> None:
    """Copie les images dans les dossiers train/validation/test."""
    target_dir = OUT_DIR / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def main() -> None:
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Nettoyage et tri du dataset")
    print("=" * 60)

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
            f"{class_dir.name:12} | Total: {total:4d} "
            f"| Train: {len(train_files):4d} "
            f"| Val: {len(val_files):3d} "
            f"| Test: {len(test_files):3d}"
        )

    print("=" * 60)
    print(f"Dataset nettoyé : {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 3.6 Utiliser le script de nettoyage

```powershell
# Depuis le répertoire yolov5 avec le venv activé
python scripts/clean_dataset.py
```

Le script :
- Détecte les images floues
- Les supprime
- Distribue les images propres : 70% train, 20% validation, 10% test
- Affiche un résumé par espèce

---

## 4. Organisation des dossiers

### 4.1 Structure finale pour la classification

Une fois nettoyé, votre dataset doit avoir cette structure :

```
dataset_oiseaux/
├── train/
│   ├── moineau/
│   │   ├── oiseau_001.jpg
│   │   ├── oiseau_002.jpg
│   │   └── ...
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

**Important :** Chaque sous-dossier porte le nom d'une classe (espèce d'oiseau).

### 4.2 Vérifier la structure

Après le nettoyage, vérifiez la structure avec PowerShell :

```powershell
# Compter les images par espèce
Get-ChildItem -Path "C:\Users\yanni\Desktop\Yolo\dataset_oiseaux\train\" -Directory | `
  ForEach-Object { 
    $count = (Get-ChildItem $_.FullName -File).Count
    Write-Host "$($_.Name): $count images"
  }
```

### 4.3 Répartition équilibrée

Assurez-vous que chaque espèce a un nombre d'images similaire :

- **Cible :** 400-500 images d'entraînement par espèce
- **Minimum acceptable :** 300 images par espèce
- **Écart maximum entre espèces :** moins de 30%

Si une espèce a beaucoup moins d'images, envisagez d'augmenter les données ou d'exclure les espèces très minoritaires.

---

## 5. Entraînement du modèle

### 5.1 Différentes architectures disponibles

YOLOv5 propose plusieurs tailles de modèles pour la classification :

| Modèle | Taille | Vitesse | Précision | Recommandé pour |
|--------|--------|---------|-----------|-----------------|
| `yolov5n-cls.pt` | Très petit | Très rapide | Bonne | Webcam en temps réel |
| `yolov5s-cls.pt` | Petit | Rapide | Très bonne | **Recommandé pour ce projet** |
| `yolov5m-cls.pt` | Moyen | Modéré | Excellente | Serveur CPU/GPU |
| `yolov5l-cls.pt` | Grand | Lent | Excellente | Serveurs puissants |

### 5.2 Paramètres d'entraînement expliqués

| Paramètre | Valeur | Explication |
|-----------|--------|-------------|
| `--model` | `yolov5s-cls.pt` | Modèle de départ (poids pré-entraînés) |
| `--data` | `dataset_oiseaux` | Dossier contenant les données (train/validation) |
| `--epochs` | `20-50` | Nombre de passages sur l'ensemble d'entraînement |
| `--img` | `224` | Taille des images (224x224 pixels pour classification) |
| `--batch` | `32` | Nombre d'images traitées simultanément |
| `--device` | `0` | GPU à utiliser (0 = GPU 0, ou `cpu` pour CPU) |

### 5.3 Lancer l'entraînement

Depuis le répertoire yolov5 avec le venv activé :

```powershell
python classify/train.py `
  --model yolov5s-cls.pt `
  --data C:\Users\yanni\Desktop\Yolo\dataset_oiseaux `
  --epochs 30 `
  --img 224 `
  --batch 32 `
  --device 0
```

**Explications des arguments :**
- `--model yolov5s-cls.pt` : modèle petit (bon compromis vitesse/précision)
- `--data` : chemin complet vers votre dossier dataset
- `--epochs 30` : 30 passages sur les données (augmentez à 50-100 si le dataset est petit)
- `--img 224` : format standard pour la classification
- `--batch 32` : traitez 32 images à la fois (réduisez à 16 si erreur mémoire GPU)
- `--device 0` : utilisez le GPU 0 (changez à `cpu` si pas de GPU)

### 5.4 Suivi de l'entraînement

Pendant l'entraînement, vous verrez :

```
Epoch  GPU_mem  img_size   loss   top1   top5  val_loss  top1   top5
  1/30    2.5G      224    2.45   40.2   15.3    1.98    45.1   12.5
  2/30    2.5G      224    2.12   52.3   22.1    1.67    58.2   19.3
  ...
 30/30    2.5G      224    0.45   92.1    3.2    0.52    89.5    5.1
```

**Interprétation :**
- `loss` : erreur d'entraînement (doit diminuer)
- `top1` : pourcentage de bonnes classifications (doit augmenter)
- `val_loss` : erreur de validation
- Plus la `top1` se rapproche de 100%, mieux c'est

### 5.5 Où sont sauvés les résultats

Les modèles entraînés sont sauvés dans :

```
runs/train-cls/exp/
├── weights/
│   ├── best.pt        (meilleur modèle)
│   └── last.pt        (dernier modèle)
├── results.csv        (historique des métriques)
└── results.png        (graphiques des performances)
```

---

## 6. Validation et test

### 6.1 Évaluer le modèle sur les données de validation

Après l'entraînement, validez le modèle :

```powershell
python classify/val.py `
  --data C:\Users\yanni\Desktop\Yolo\dataset_oiseaux `
  --weights runs/train-cls/exp/weights/best.pt `
  --img 224 `
  --batch 32
```

Cela affichera la **matrice de confusion** et les métriques par espèce.

### 6.2 Tester sur de nouvelles images

Testez sur des images non vues pendant l'entraînement :

```powershell
python classify/predict.py `
  --weights runs/train-cls/exp/weights/best.pt `
  --source C:\Users\yanni\Desktop\Yolo\dataset_oiseaux\test\moineau\oiseau_001.jpg `
  --img 224
```

Le modèle affichera la classe prédite et la confiance.

### 6.3 Interprétation des résultats

Après la validation, vérifiez :

| Métrique | Acceptable | Bon | Excellent |
|----------|-----------|------|----------|
| Top-1 Accuracy | > 80% | > 90% | > 95% |
| Top-5 Accuracy | > 95% | > 98% | > 99% |
| Loss de validation | < 1.0 | < 0.5 | < 0.2 |

Si les résultats ne sont pas satisfaisants :
1. Augmentez le nombre d'epochs
2. Augmentez la taille du batch
3. Améliorez la qualité du dataset
4. Essayez un modèle plus grand (yolov5m-cls)

### 6.4 Test final sur 1000 images

Pour le test de production final :

1. Créez un ensemble de **1000 images variées** :
   - Images des 6 espèces cibles
   - Images d'autres espèces (pour tester les faux positifs)
   - Images non vues pendant l'entraînement

2. Testez le modèle :

```powershell
python classify/predict.py `
  --weights runs/train-cls/exp/weights/best.pt `
  --source C:\chemin\vers\test_final_1000_images\ `
  --img 224
```

3. Mesurez :
   - Taux de bonne identification pour les 6 espèces
   - Taux de faux positifs sur les autres espèces
   - Confiance moyenne par espèce

---

## 7. Pipeline temps réel

### 7.1 Architecture complète

Le système final fonctionne ainsi :

```
Capture image
    ↓
Détection de l'oiseau (localisation)
    ↓
Recadrage de la zone détectée
    ↓
Classification de l'espèce
    ↓
Confiance > seuil ?
    ├─ OUI → Lecture du son correspondant
    └─ NON → Pas de son
    ↓
Enregistrement du résultat
```

### 7.2 Configuration des espèces et sons

Créez un fichier `config/birds_sounds.yaml` :

```yaml
# Mapping espèces → sons
moineau:
  - sounds/moineau/son_moineau1.mp3
  - sounds/moineau/son_moineau2.mp3
  - sounds/moineau/son_moineau3.mp3

corbeau:
  - sounds/corbeau/croassement1.mp3
  - sounds/corbeau/croassement2.mp3

pigeon:
  - sounds/pigeon/roucoulement1.mp3
  - sounds/pigeon/roucoulement2.mp3

heron:
  - sounds/heron/cri_heron1.mp3
  - sounds/heron/cri_heron2.mp3

merle:
  - sounds/merle/chant_merle1.mp3
  - sounds/merle/chant_merle2.mp3

aigrette:
  - sounds/aigrette/cri_aigrette1.mp3
  - sounds/aigrette/cri_aigrette2.mp3

# Confiance minimale pour jouer un son
confidence_threshold: 0.75

# Délai anti-répétition (secondes)
repeat_delay: 3.0
```

### 7.3 Organisation des dossiers pour le déploiement

```
project/
├── dataset_raw/                    (données brutes, jamais modifiées)
├── dataset_oiseaux/                (données nettoyées)
├── models/
│   ├── best.pt                     (modèle entraîné)
│   └── last.pt                     (dernier modèle)
├── sounds/
│   ├── moineau/
│   │   ├── son_moineau1.mp3
│   │   └── son_moineau2.mp3
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
├── config/
│   └── birds_sounds.yaml           (configuration espèces ↔ sons)
├── scripts/
│   ├── clean_dataset.py            (nettoyage)
│   └── inference.py                (inférence temps réel)
└── app/
    ├── main.py                     (application principale)
    └── requirements.txt
```

### 7.4 Script d'inférence simple

Créez `app/inference.py` :

```python
import cv2
import random
import torch
from pathlib import Path
import pygame
import yaml
from typing import Optional

class BirdDetectionApp:
    def __init__(self, weights_path: str, config_path: str):
        """
        Initialise l'application.
        
        Args:
            weights_path: chemin vers best.pt
            config_path: chemin vers birds_sounds.yaml
        """
        # Charger le modèle
        self.model = torch.hub.load("ultralytics/yolov5", "custom", 
                                     path=weights_path, force_reload=False)
        
        # Charger la configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = list(self.config.keys())[:-2]  # Exclure les paramètres
        self.threshold = self.config.get("confidence_threshold", 0.75)
        self.repeat_delay = self.config.get("repeat_delay", 3.0)
        
        # Initialiser pygame pour l'audio
        pygame.mixer.init()
        
        # Historique des derniers sons joués
        self.last_played = {}
    
    def predict(self, image_path: str) -> Optional[str]:
        """
        Prédit la classe d'un oiseau dans une image.
        
        Args:
            image_path: chemin vers l'image
        
        Returns:
            nom de l'espèce détectée, ou None
        """
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur : impossible de charger {image_path}")
            return None
        
        # Inférence
        results = self.model(img)
        predictions = results.pandas().xyxy[0]
        
        if len(predictions) == 0:
            print("Aucun oiseau détecté")
            return None
        
        # Prendre la détection avec la plus haute confiance
        best_pred = predictions.iloc[0]
        class_id = int(best_pred["class"])
        confidence = float(best_pred["confidence"])
        predicted_class = self.class_names[class_id]
        
        print(f"Espèce détectée : {predicted_class}")
        print(f"Confiance : {confidence:.2%}")
        
        return predicted_class if confidence >= self.threshold else None
    
    def play_sound(self, species: str) -> None:
        """
        Joue un son aléatoire pour une espèce.
        
        Args:
            species: nom de l'espèce
        """
        if species not in self.config:
            print(f"Espèce inconnue : {species}")
            return
        
        sounds = self.config[species]
        sound_file = random.choice(sounds)
        
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            print(f"Lecture : {sound_file}")
        except Exception as e:
            print(f"Erreur audio : {e}")
    
    def process_image(self, image_path: str) -> None:
        """
        Traite une image : détecte, classifie et joue le son.
        
        Args:
            image_path: chemin vers l'image
        """
        species = self.predict(image_path)
        
        if species:
            self.play_sound(species)
        else:
            print("Pas de son joué (confiance insuffisante ou espèce inconnue)")

# Utilisation
if __name__ == "__main__":
    app = BirdDetectionApp(
        weights_path=r"C:\Users\yanni\Desktop\Yolo\models\best.pt",
        config_path=r"C:\Users\yanni\Desktop\Yolo\config\birds_sounds.yaml"
    )
    
    # Tester sur une image
    app.process_image(r"C:\chemin\vers\image_test.jpg")
```

---

## 8. Checklist avant lancement

Avant de lancer l'entraînement, vérifiez tous ces points :

### Environnement
- [ ] Python 3.8+ installé
- [ ] PyTorch >= 1.8 installé (`python -c "import torch"`)
- [ ] Venv activé (`.venv\Scripts\Activate.ps1`)
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] GPU détecté (optionnel : `python -c "import torch; print(torch.cuda.is_available())"`)

### Dataset
- [ ] Dossier `dataset_raw/` contient les images brutes
- [ ] Dossier `dataset_oiseaux/` avec structure train/validation/test
- [ ] Chaque espèce dans ses propres sous-dossiers
- [ ] Aucune image floue, trop sombre ou cassée
- [ ] Doublons supprimés
- [ ] Nombre d'images équilibré par espèce (écart < 30%)
- [ ] Minimum 300-400 images par espèce en train

### Dossiers et fichiers
- [ ] `dataset_oiseaux/train/` existe et contient des images
- [ ] `dataset_oiseaux/validation/` existe et contient des images
- [ ] `dataset_oiseaux/test/` existe (optionnel mais recommandé)

### Noms de classes
- [ ] Tous les noms de classes sont en minuscules
- [ ] Aucun accent ni caractère spécial dans les noms
- [ ] Les 6 classes sont : moineau, corbeau, pigeon, heron, merle, aigrette

### Configuration d'entraînement
- [ ] Paramètres d'entraînement adaptés à votre matériel
- [ ] Batch size approprié (réduisez si erreur mémoire)
- [ ] Nombre d'epochs suffisant (30-50 minimum)
- [ ] Modèle de départ choisi (yolov5s-cls recommandé)

### Sons et configuration (optionnel au départ)
- [ ] Dossier `sounds/` créé
- [ ] Sous-dossiers par espèce
- [ ] Fichiers audio en format compatible (.mp3, .wav)
- [ ] Fichier `config/birds_sounds.yaml` créé

---

## Résumé des étapes clés

### Phase 1 : Préparation (1-2 heures)
1. Clonez/préparez le dépôt YOLOv5
2. Activez l'environnement virtuel
3. Installez les dépendances

### Phase 2 : Données (2-4 heures)
4. Nettoyez manuellement un échantillon du dataset
5. Exécutez le script de nettoyage automatique
6. Vérifiez la structure et l'équilibre

### Phase 3 : Entraînement (1-8 heures selon GPU)
7. Lancez l'entraînement avec les paramètres appropriés
8. Suivez les métriques (loss, accuracy)

### Phase 4 : Validation (30 minutes)
9. Validez sur le dataset de validation
10. Évaluez les résultats
11. Ajustez si nécessaire

### Phase 5 : Déploiement (optionnel)
12. Organisez la structure finale
13. Créez les scripts d'inférence
14. Testez sur nouvelles images

---

## Commandes rapides de référence

```powershell
# Activation
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
.\.venv\Scripts\Activate.ps1

# Installation
python -m pip install -r requirements.txt

# Nettoyage
python scripts/clean_dataset.py

# Entraînement
python classify/train.py --model yolov5s-cls.pt --data C:\Users\yanni\Desktop\Yolo\dataset_oiseaux --epochs 30 --img 224 --batch 32

# Validation
python classify/val.py --data C:\Users\yanni\Desktop\Yolo\dataset_oiseaux --weights runs/train-cls/exp/weights/best.pt --img 224

# Prédiction
python classify/predict.py --weights runs/train-cls/exp/weights/best.pt --source C:\chemin\vers\image.jpg
```

---

## Troubleshooting

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Réinstallez PyTorch : `pip install torch` |
| Erreur CUDA : `CUDA out of memory` | Réduisez le batch size : `--batch 16` |
| Images floues non détectées | Abaissez `BLUR_THRESHOLD` dans le script |
| Modèle n'apprend pas (loss élevée) | Augmentez epochs, vérifiez les labels |
| Accuracy faible | Améliorez le dataset, essayez un plus grand modèle |

---

## Ressources supplémentaires

- [Documentation YOLOv5](https://docs.ultralytics.com/yolov5/)
- [Notebook tutorial](../tutorial.ipynb)
- [Guide classification](../classify/tutorial.ipynb)
- [GUIDE_OISEAUX.md](../GUIDE_OISEAUX.md) - Documentation détaillée du projet

---

**Dernière mise à jour :** 11 mai 2026  
**Auteur :** Guide automatisé basé sur GUIDE_OISEAUX.md
