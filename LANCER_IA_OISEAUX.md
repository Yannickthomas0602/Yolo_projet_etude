# Lancement de l'IA oiseaux

Ce document sert de procédure courte pour lancer le modèle d'oiseaux, tester une image et comprendre le résultat métier.

Le projet actuel repose sur 4 classes :
- balbuzard
- heron
- cormoran
- mouette_goeland

Le modèle de référence entraîné se trouve dans runs/train-cls/exp4/weights/best.pt.

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
  --weights runs/train-cls/exp4/weights/best.pt `
  --source C:\Users\yanni\Desktop\Yolo\yolov5\dataset_oiseaux\train\cormoran\pixabay_91570.jpg `
  --device 0 `
  --bdd-thres 0.60 `
  --uncertainty-thres 0.30
```

Résultat attendu :
- BDD si la confiance top-1 est suffisante,
- INCERTITUDE si la confiance est intermédiaire, avec un seuil d'incertitude à 30 % par défaut,
- HORS_BDD si la confiance est trop faible.

Les résultats annotés sont enregistrés dans runs/predict-cls/exp*/.

## 3. Tester un dossier entier

Pour vérifier plusieurs images d'un coup :

```powershell
python classify/predict.py `
  --weights runs/train-cls/exp4/weights/best.pt `
  --source C:\Users\yanni\Desktop\Yolo\yolov5\dataset_oiseaux\test `
  --device 0 `
  --bdd-thres 0.60 `
  --uncertainty-thres 0.30
```

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
