# Lancement de l'IA oiseaux

Ce document sert de procédure courte pour lancer le modèle d'oiseaux, tester une image et comprendre le résultat métier.

Le projet actuel repose sur 4 classes :
- balbuzard
- heron
- cormoran
- mouette_goeland

Le modèle de référence entraîné se trouve dans runs/train-cls/exp_retrain/weights/best.pt.

## 1. Préparer le terminal

Depuis PowerShell, place-toi dans le dépôt puis active le venv :

```powershell
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
.\.venv\Scripts\Activate.ps1
```

Vérifie rapidement que PyTorch voit le GPU :

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 2. Tester une image

Commande type sur une image unique :

```powershell
python classify/predict.py `
  --weights runs/train-cls/exp_retrain/weights/best.pt `
  --source C:\Users\yanni\Desktop\Yolo\yolov5\dataset_oiseaux\train\cormoran\pixabay_91570.jpg `
  --device 0 `
  --bdd-thres 0.60 `
  --uncertainty-thres 0.50
```

Résultat attendu :
- BDD si la confiance top-1 est supérieure ou égale à 60 %, 
- INCERTITUDE si la confiance top-1 est entre 50 % et 60 %,
- HORS_BDD si la confiance top-1 est inférieure à 50 %.

Important : dans [analyse_oiseaux.py](analyse_oiseaux.py), la décision métier est basée uniquement sur le score top-1.

Les résultats annotés sont enregistrés dans runs/predict-cls/exp*/.

## 3. Tester un dossier entier

Pour vérifier plusieurs images d'un coup :

```powershell
python classify/predict.py `
  --weights runs/train-cls/exp_retrain/weights/best.pt `
  --source C:\Users\yanni\Desktop\Yolo\yolov5\dataset_oiseaux\test `
  --device 0 `
  --bdd-thres 0.60 `
  --uncertainty-thres 0.50
```

Si ton dossier contient des sous-dossiers de classes, tu peux aussi lancer [analyse_oiseaux.py](analyse_oiseaux.py) pour analyser toute l'arborescence d'un coup et afficher le graphique automatiquement.

Note : en mode dossier, `analyse_oiseaux.py` n'enregistre pas les images annotées une par une (utilise `--nosave` pour accélérer l'analyse) ; il génère un résumé JSON et des graphiques professionnels dans le dossier `results/`.

Si tu veux utiliser la camera de l'ordinateur et comprendre où sont rangées les images selon leur statut, lis aussi [MODE_CAMERA_OISEAUX.md](MODE_CAMERA_OISEAUX.md).

## 3.2 Synchroniser automatiquement sur Azure (optionnel)

Le script [analyse_oiseaux.py](analyse_oiseaux.py) peut aussi pousser chaque détection vers Azure Blob Storage et Azure IoT Hub si les variables d'environnement suivantes sont définies. Le script charge automatiquement un fichier local [`.env`](.env) s'il est présent à la racine du projet.

- `AZURE_STORAGE_CONN`
- `AZURE_IOT_HUB_CONN`
- `AZURE_BLOB_CONTAINER` (défaut : `archives-photos`)
- `AZURE_APPAREIL` (défaut : `Bassin_01`)

Exemple PowerShell :

```powershell
$env:AZURE_STORAGE_CONN = "DefaultEndpointsProtocol=https;..."
$env:AZURE_IOT_HUB_CONN = "HostName=...;DeviceId=...;SharedAccessKey=..."
$env:AZURE_BLOB_CONTAINER = "archives-photos"
$env:AZURE_APPAREIL = "Bassin_01"
python analyse_oiseaux.py
```

Le téléversement utilise l'image déjà renommée localement, puis envoie un message IoT standardisé avec la classe détectée, le score et le statut métier.

## 3.1 Ajouter une classe "autre" et ré-entraîner (optionnel)

Si tu veux que le modèle apprenne explicitement une classe `autre` pour représenter les espèces hors BDD, suis ces étapes :

1. Rassemble des images d'espèces hors BDD dans un dossier, par exemple `C:\Users\yanni\Desktop\autres_images`.
2. Lance le script d'import (il va copier et répartir les images dans `dataset_oiseaux`):

```powershell
Set-Location C:\Users\yanni\Desktop\Yolo\yolov5
python scripts\add_autre.py --source C:\Users\yanni\Desktop\autres_images
```

3. Vérifie les dossiers créés : `dataset_oiseaux\train\autre`, `dataset_oiseaux\validation\autre`, `dataset_oiseaux\test\autre`.
4. Lance l'entraînement (GPU recommandé) :

```powershell
python train.py `
  --model yolov5s-cls.pt `
  --data dataset_oiseaux `
  --epochs 30 `
  --img 224 `
  --batch 32 `
  --device 0
```

Après entraînement, remplace `--weights` dans tes commandes d'analyse par le nouveau jeu de poids généré dans `runs/train-cls/exp*/weights/best.pt`.

## 4. Relancer l'entraînement

Si tu modifies le dataset et veux refaire l'entraînement :

```powershell
python classify/train.py `
  --model yolov5s-cls.pt `
  --data dataset_oiseaux `
  --epochs 30 `
  --img 224 `
  --batch 32 `
  --device 0
```

Les poids seront enregistrés dans runs/train-cls/exp*/weights/.

## 5. Où regarder en cas de doute

- Démarche complète : [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- Détails d'installation et d'entraînement : [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- Historique des décisions : [JOURNAL_PROJET_OISEAUX.md](JOURNAL_PROJET_OISEAUX.md)
- Logique métier BDD / INCERTITUDE / HORS_BDD : [classify/predict.py](classify/predict.py)
 - Recherche visuelle (vectorielle) : [MODE_VECTORIEL.md](MODE_VECTORIEL.md)
