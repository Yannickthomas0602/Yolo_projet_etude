#!/usr/bin/env python3
"""Script de validation du modèle exp_retrain sur le dataset test."""

from analyse_oiseaux import analyze_folder
from pathlib import Path

print("\n" + "="*70)
print("VALIDATION DU MODÈLE RÉENTRAÎNÉ (exp_retrain)")
print("="*70 + "\n")

analyze_folder(Path('dataset_oiseaux/test'))

print("\n" + "="*70)
print("✅ VALIDATION TERMINÉE")
print("="*70)
