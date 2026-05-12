#!/usr/bin/env python3
"""
Wrapper pour entraîner le classifieur YOLOv5 sur dataset_oiseaux sans passer par le système de téléchargement.
"""
import sys
from pathlib import Path

# Ajoute le dossier yolov5 au chemin
yolov5_root = Path(__file__).resolve().parent
sys.path.insert(0, str(yolov5_root))

from classify.train import parse_opt, main as train_main

if __name__ == '__main__':
    # Arguments par défaut
    opt = parse_opt()
    opt.model = 'yolov5s-cls.pt'
    opt.data = str(yolov5_root / 'dataset_oiseaux')  # Utilise directement le dossier, pas le YAML
    opt.epochs = 30
    opt.img = 224
    opt.batch_size = 32
    opt.device = '0'
    opt.project = str(yolov5_root / 'runs' / 'train-cls')
    opt.name = 'exp_retrain'
    opt.exist_ok = True
    
    print(f"Entraînement avec:")
    print(f"  Model: {opt.model}")
    print(f"  Data: {opt.data}")
    print(f"  Epochs: {opt.epochs}")
    print(f"  Device: {opt.device}")
    
    train_main(opt)
