# ============================================================================
# ANALYSE OISEAUX - Script d'analyse des prédictions YOLOv5 pour classification d'oiseaux
# Ce script lance des prédictions avec un modèle YOLOv5 entraîné et analyse les résultats
# ============================================================================

from __future__ import annotations

# ============================================================================
# IMPORTS - Modules standards et dépendances externes
# ============================================================================
import hashlib              # Pour générer des identifiants uniques (SHA1)
import importlib.util       # Pour vérifier la disponibilité des modules
import os                   # Pour les opérations système
import re                   # Pour les expressions régulières
import shutil               # Pour copier les images analysées
import subprocess           # Pour exécuter des processus externes (YOLOv5)
import sys                  # Pour l'accès aux arguments et l'interpréteur Python
import unicodedata          # Pour normaliser les caractères Unicode
import random               # Pour sélectionner aléatoirement un fichier audio
from datetime import datetime  # Pour horodater les enregistrements
from collections import Counter  # Pour compter les occurrences
from dataclasses import dataclass  # Pour créer des classes structurées
from pathlib import Path    # Pour gérer les chemins de fichier
from typing import Dict, List, Optional  # Pour les annotations de type
import json                 # Pour manipuler du JSON (sauvegarde des résultats)
import argparse            # Pour parser les arguments (non utilisé actuellement)
import tempfile            # Pour créer des répertoires temporaires
import threading
import time
import math
from importlib import import_module

# ============================================================================
# CONSTANTES - Chemins et paramètres de configuration
# ============================================================================

# Répertoire racine du projet YOLOv5
ROOT = Path(__file__).resolve().parent

# Chemin vers le modèle de poids entraîné (YOLOv5 classification)
WEIGHTS = ROOT / "runs" / "train-cls" / "exp_retrain" / "weights" / "best.pt"

# Script YOLOv5 qui lance les prédictions de classification
PREDICT_SCRIPT = ROOT / "classify" / "predict.py"

# Répertoire où seront sauvegardés tous les résultats (graphiques, JSON, etc.)
RESULTS_DIR = ROOT / "results"

# Répertoire où seront copiées les images analysées par statut
ENREGISTREMENTS_DIR = ROOT / "enregistrements"

# Répertoire contenant les fichiers audio des cris d'oiseaux
AUDIO_DIR = ROOT / "cri_predateur_ou_detresse"

# Extensions de fichiers image supportées
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Fichier audio d'alerte (doute) - joué en cas d'INCERTITUDE
AUDIO_ALERT = AUDIO_DIR / "canon.mp3"

# Extensions audio supportées
AUDIO_EXTENSIONS = {".mp3"}

# Répertoire pour l'index vectoriel (FAISS + mapping)
VECTOR_DIR = ROOT / "vectors"
# Seuil de similarité (cosine approximé par inner product sur vecteurs normalisés)
VECTOR_SIMILARITY_THRESH = 0.30

# ============================================================================
# SEUILS DE CONFIANCE - Pour classifier les prédictions
# ============================================================================
# Les prédictions sont catégorisées en 3 statuts selon le score de confiance (top-1 uniquement):
#   - BDD : confiance >= BDD_THRES (60%)           → classe reconnue avec certitude
#   - INCERTITUDE : INCERTITUDE_THRES <= confiance < BDD_THRES (50%-60%) → doute
#   - HORS_BDD : confiance < INCERTITUDE_THRES (<50%)  → classe inconnue/hors dataset
BDD_THRES = 0.60
INCERTITUDE_THRES = 0.50



# ============================================================================
# GESTION DES DÉPENDANCES - Installation automatique si manquantes
# ============================================================================

def ensure_dependency(module_name: str, pip_name: str | None = None) -> None:
    """
    Vérifie si un module Python est installé, sinon l'installe automatiquement.
    
    Args:
        module_name: Nom du module à importer (ex: 'matplotlib')
        pip_name: Nom du package pip (optionnel, utilisé si différent du module_name)
    """
    # Si le module est déjà installé, on ne fait rien
    if importlib.util.find_spec(module_name) is not None:
        return

    # Utilise le nom pip fourni ou se replie sur le nom du module
    package_name = pip_name or module_name
    print(f"[INFO] Module manquant détecté: {module_name}. Installation en cours avec pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def ensure_runtime_dependencies() -> None:
    """Installe les dépendances requises pour le runtime (matplotlib, tqdm, colorama)."""
    for module_name, pip_name in (("matplotlib", None), ("tqdm", None), ("colorama", None), ("cv2", "opencv-python")):
        ensure_dependency(module_name, pip_name)


# Vérifie et installe les dépendances avant de les importer
ensure_runtime_dependencies()

# ============================================================================
# IMPORTS TARDIFS - Dépendances externes garanties d'être disponibles
# ============================================================================
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from tqdm import tqdm  # noqa: E402
import cv2  # noqa: E402

try:
    # Tente d'importer colorama pour les couleurs en console
    from colorama import Fore, Style, init as colorama_init  # noqa: E402
    colorama_init(autoreset=True)
except Exception:  # pragma: no cover - fallback si colorama échoue
    # Fallback classes si colorama n'est pas disponible (mode dégradé)
    class _FallbackColor:
        """Couleurs vides pour remplacer colorama en cas d'erreur."""
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""

    class _FallbackStyle:
        """Styles vides pour remplacer colorama en cas d'erreur."""
        BRIGHT = ""
        NORMAL = ""
        RESET_ALL = ""

    Fore = _FallbackColor()
    Style = _FallbackStyle()

# ============================================================================
# CLASSE DATACLASS - Structure pour stocker une prédiction YOLOv5
# ============================================================================

@dataclass
class PredictionRecord:
    """
    Représente une seule prédiction de classification d'image YOLOv5.
    Contient tous les scores de confiance et le statut post-traitement.
    """
    image_path: Path              # Chemin vers l'image analysée
    top1_class: str              # Classe prédite avec la plus haute confiance
    top1_score: float            # Score de confiance pour la top-1 classe (0-1)
    status: str                  # Statut: BDD, INCERTITUDE ou HORS_BDD
    class_scores: Dict[str, float]  # Tous les scores par classe
    raw_line: str                # Ligne brute du output YOLOv5

# ============================================================================
# EXPRESSION RÉGULIÈRE - Pattern pour parser les lignes YOLOv5
# ============================================================================
PREDICTION_LINE_RE = re.compile(
    r"^image\s+\d+/\d+\s+(?P<path>.+):\s+"
    r"(?P<size>\d+x\d+)\s+"
    r"\[(?P<status>BDD|INCERTITUDE|HORS_BDD)\]\s+"
    r"(?P<top1_class>.+?)\s+(?P<top1_score>[0-9]*\.?[0-9]+),\s+"
    r"(?P<rest>.+),\s+(?P<time>[0-9]*\.?[0-9]+)ms$"
)


# ============================================================================
# FONCTIONS UTILITAIRES - Helpers pour affichage et traitement des données
# ============================================================================

def console_text(text: str, color: str = "", bright: bool = False) -> str:
    """
    Formate du texte avec couleur et style pour l'affichage en console.
    
    Args:
        text: Texte à formater
        color: Couleur colorama (ex: Fore.RED)
        bright: True pour un style BRIGHT (texte plus clair)
    
    Returns:
        Texte formaté avec codes ANSI de couleur
    """
    prefix = f"{Style.BRIGHT if bright else ''}{color}"
    return f"{prefix}{text}{Style.RESET_ALL}"


def normalize_label(value: str) -> str:
    """
    Normalise un label pour la comparaison robuste entre dossier et prédiction.
    Supprime accents, espaces, caractères spéciaux, convertit en minuscules.
    
    Args:
        value: Label brut (ex: 'Rouge-Gorge')
    
    Returns:
        Label normalisé (ex: 'rouge_gorge')
    """
    # Normalise les accents (décompose les caractères accentués)
    normalized = unicodedata.normalize("NFKD", value)
    # Supprime les diacritiques (accents)
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    # Minuscules et supprime les espaces
    normalized = normalized.lower().strip()
    # Remplace tirets et espaces par underscores
    normalized = normalized.replace("-", "_").replace(" ", "_")
    # Supprime tous les caractères non alphanumériques sauf underscore
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    # Supprime les underscores multiples et aux extrémités
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def find_image_files(folder: Path) -> List[Path]:
    """
    Trouve toutes les images supportées dans un dossier, récursivement.
    
    Args:
        folder: Chemin du dossier à explorer
    
    Returns:
        Liste triée des chemins vers les images trouvées
    """
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_prediction_line(line: str) -> PredictionRecord | None:
    """
    Parse une ligne de output YOLOv5 en une structure PredictionRecord.
    
    Args:
        line: Ligne de log YOLOv5 (ex: "image 1/10 /path/to/img.jpg: 400x300 [BDD] robin 0.85, ...")
    
    Returns:
        PredictionRecord si le parsing réussit, None sinon
    """
    # Teste la ligne contre le pattern regex
    match = PREDICTION_LINE_RE.match(line.strip())
    if match is None:
        return None

    # Extrait les groupes du regex
    image_path = Path(match.group("path"))
    top1_class = match.group("top1_class").strip()
    top1_score = float(match.group("top1_score"))
    status = match.group("status")

    # Extrait tous les scores par classe depuis la partie "rest"
    class_scores: Dict[str, float] = {top1_class: top1_score}
    rest = match.group("rest").strip()
    # Parse le format "classe1 0.85, classe2 0.10, ..." 
    for chunk in rest.split(", "):
        if not chunk:
            continue
        name, score_text = chunk.rsplit(" ", 1)
        class_scores[name.strip()] = float(score_text)

    return PredictionRecord(
        image_path=image_path,
        top1_class=top1_class,
        top1_score=top1_score,
        status=status,
        class_scores=class_scores,
        raw_line=line.strip(),
    )


def build_run_name(source: Path) -> str:
    """
    Génère un nom unique court pour les outputs YOLOv5.
    Utilise le nom du fichier normalisé + hash SHA1 du chemin complet.
    
    Args:
        source: Chemin du fichier/dossier
    
    Returns:
        Nom unique (ex: 'rouge_gorge_a1b2c3d4e5')
    """
    # Utilise le stem (nom sans extension) ou le nom du dossier
    base_name = normalize_label(source.stem if source.is_file() else source.name) or "analysis"
    # Génère un hash SHA1 court du chemin absolu pour l'unicité
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{base_name}_{digest}"


def get_recording_subfolder(record: PredictionRecord) -> str:
    """
    Détermine le sous-dossier de sauvegarde en fonction du statut métier.

    Args:
        record: Résultat d'analyse YOLOv5

    Returns:
        Nom du sous-dossier cible
    """
    if record.status == "BDD":
        return normalize_label(record.top1_class) or "bdd"
    if record.status == "INCERTITUDE":
        return "incertitude"
    return "autre"


def build_recording_name(source_path: Path, record: PredictionRecord) -> str:
    """
    Génère un nom de fichier unique pour l'image enregistrée.

    Args:
        source_path: Image analysée
        record: Résultat d'analyse associé

    Returns:
        Nom de fichier unique au format JPG
    """
    source_base = normalize_label(source_path.stem) or "capture"
    status_base = get_recording_subfolder(record)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    digest = hashlib.sha1(
        f"{source_path.resolve()}|{record.status}|{record.top1_class}|{timestamp}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{source_base}_{status_base}_{timestamp}_{digest}.jpg"


def save_analyzed_image(source_path: Path, record: PredictionRecord) -> Optional[Path]:
    """
    Copie l'image analysée dans un dossier dédié au statut détecté.

    Args:
        source_path: Image analysée
        record: Résultat d'analyse associé

    Returns:
        Chemin vers l'image copiée, ou None si la copie a échoué
    """
    if not source_path.exists() or not source_path.is_file():
        print(console_text(f"[ENREGISTREMENT] Image introuvable: {source_path}", Fore.YELLOW, bright=True))
        return None

    target_dir = ENREGISTREMENTS_DIR / get_recording_subfolder(record)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / build_recording_name(source_path, record)

    try:
        shutil.copy2(source_path, target_path)
    except Exception as exc:
        print(console_text(f"[ENREGISTREMENT] Impossible de copier l'image: {exc}", Fore.RED, bright=True))
        return None

    print(console_text(f"[ENREGISTREMENT] Image sauvegardée dans: {target_path}", Fore.CYAN, bright=True))
    return target_path


def open_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Ouvre la camera de l'ordinateur.

    Args:
        camera_index: Index de la camera à ouvrir

    Returns:
        Instance VideoCapture ouverte

    Raises:
        RuntimeError: Si la camera ne peut pas être ouverte
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW") else 0
    camera = cv2.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera {camera_index}.")
    return camera


def capture_frame_to_tempfile(frame, temp_dir: Path) -> Path:
    """
    Enregistre une frame camera dans un fichier temporaire JPEG.

    Args:
        frame: Image OpenCV
        temp_dir: Répertoire temporaire

    Returns:
        Chemin du fichier image temporaire
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_name = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    frame_path = temp_dir / frame_name
    success = cv2.imwrite(str(frame_path), frame)
    if not success:
        raise RuntimeError("Impossible d'enregistrer la frame camera dans un fichier temporaire.")
    return frame_path


# ============================================================================
# GESTION AUDIO - Recherche et lecture des fichiers audio des oiseaux
# ============================================================================

# Mappage des classes du modèle (normalisées) vers les dossiers audio
CLASS_TO_AUDIO_FOLDER = {
    "balbuzard": "Balbuzard",
    "heron": "Heron",
    "cormoran": "Cormoran",
    "mouette_goeland": "Goeland-mouette",
}


def find_audio_file(bird_class: str) -> Optional[Path]:
    """
    Recherche et retourne un fichier audio aléatoire correspondant à un oiseau détecté.
    
    Args:
        bird_class: Nom de la classe détectée (doit être normalisé, ex: 'balbuzard')
    
    Returns:
        Chemin vers un fichier MP3 sélectionné aléatoirement, ou None si aucun fichier trouvé
    """
    # Normalise le nom de classe pour la comparaison
    normalized_class = normalize_label(bird_class)
    
    # Récupère le dossier audio correspondant
    audio_folder_name = CLASS_TO_AUDIO_FOLDER.get(normalized_class)
    if audio_folder_name is None:
        print(console_text(f"[AUDIO] Classe '{bird_class}' non mappée vers un dossier audio.", Fore.YELLOW, bright=True))
        return None
    
    # Construit le chemin du dossier audio
    audio_folder = AUDIO_DIR / audio_folder_name
    
    # Vérifie que le dossier existe
    if not audio_folder.exists() or not audio_folder.is_dir():
        print(console_text(f"[AUDIO] Dossier audio introuvable pour '{bird_class}': {audio_folder}", Fore.YELLOW, bright=True))
        return None
    
    # Cherche tous les fichiers MP3 du dossier
    audio_files = [
        f for f in audio_folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    
    if not audio_files:
        print(console_text(f"[AUDIO] Aucun fichier MP3 trouvé pour '{bird_class}' dans {audio_folder}", Fore.YELLOW, bright=True))
        return None
    
    # Sélectionne aléatoirement un fichier
    selected_audio = random.choice(audio_files)
    print(console_text(f"[AUDIO] Sélectionné pour '{bird_class}': {selected_audio.name}", Fore.CYAN))
    
    return selected_audio


def play_audio(audio_path: Optional[Path]) -> bool:
    """
    Joue un fichier audio en ouvrant le lecteur système par défaut (Windows).
    La fonction est bloquante seulement si le lecteur reste ouvert.
    
    Args:
        audio_path: Chemin vers le fichier MP3 à jouer
    
    Returns:
        True si la lecture a réussi, False sinon
    """
    if audio_path is None:
        return False
    
    if not audio_path.exists():
        print(console_text(f"[AUDIO] Fichier audio introuvable: {audio_path}", Fore.RED, bright=True))
        return False
    
    try:
        # Sous Windows, os.startfile() ouvre le fichier avec l'application par défaut
        if sys.platform == "win32":
            os.startfile(str(audio_path))
            print(console_text(f"[AUDIO] Lecture lancée: {audio_path.name}", Fore.GREEN, bright=True))
            return True
        # Sous Linux, utilise 'xdg-open'
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(audio_path)])
            print(console_text(f"[AUDIO] Lecture lancée: {audio_path.name}", Fore.GREEN, bright=True))
            return True
        # Sous macOS, utilise 'open'
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(audio_path)])
            print(console_text(f"[AUDIO] Lecture lancée: {audio_path.name}", Fore.GREEN, bright=True))
            return True
        else:
            print(console_text(f"[AUDIO] Plateforme non supportée: {sys.platform}", Fore.YELLOW, bright=True))
            return False
    except Exception as e:
        print(console_text(f"[AUDIO] Erreur lors de la lecture: {e}", Fore.RED, bright=True))
        return False


def play_bird_audio(bird_class: str) -> bool:
    """
    Cherche et joue le son d'un oiseau détecté.
    Gère l'ensemble du processus: recherche du fichier, sélection aléatoire et lecture.
    
    Args:
        bird_class: Nom de la classe de l'oiseau détecté
    
    Returns:
        True si la lecture a réussi, False sinon
    """
    print(console_text(f"[AUDIO] Recherche d'un son pour '{bird_class}'...", Fore.CYAN))
    audio_file = find_audio_file(bird_class)
    if audio_file is None:
        return False
    return play_audio(audio_file)


# ============================================================================
# FONCTION PRINCIPALE D'INFÉRENCE - Exécution du modèle YOLOv5
# ============================================================================

def run_yolov5_prediction(source: Path, save_outputs: bool = True) -> List[PredictionRecord]:
    """
    Lance le classifieur YOLOv5 officiel en sous-processus et capture les prédictions.
    
    Args:
        source: Chemin vers une image ou un fichier texte listant les images
        save_outputs: Si True, sauvegarde les résultats dans RESULTS_DIR
    
    Returns:
        Liste de PredictionRecord après post-traitement des seuils
    
    Raises:
        RuntimeError: Si YOLOv5 échoue ou aucune prédiction n'est trouvée
    """
    # Construit la commande YOLOv5 à exécuter en sous-processus
    command = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--weights",
        str(WEIGHTS),              # Modèle entraîné
        "--source",
        str(source),               # Image ou liste d'images
        "--device",
        "0",                       # GPU 0 (ou CPU si pas de GPU)
        "--bdd-thres",
        "0.60",                    # Seuil BDD initial (sera recalculé)
        "--uncertainty-thres",
        "0.30",                    # Seuil incertitude initial (sera recalculé)
    ]

    # Ajoute les arguments pour sauvegarder les outputs
    if save_outputs:
        command.extend(
            [
                "--save-txt",      # Sauvegarde les résultats en texte
                "--project",
                str(RESULTS_DIR),  # Répertoire de sauvegarde
                "--name",
                build_run_name(source),  # Nom de run unique
                "--exist-ok",      # Écrase les anciens résultats
            ]
        )
    else:
        command.append("--nosave")  # Mode sans sauvegarde (pour analyse_folder temporaire)

    # Exécute YOLOv5 en sous-processus et capture stdout/stderr
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    # Récupère la sortie complète (stdout + stderr)
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    
    # Vérifie si la commande a échoué
    if completed.returncode != 0:
        print(console_text("[ERREUR] La commande YOLOv5 a échoué.", Fore.RED, bright=True))
        print(output)
        raise RuntimeError("YOLOv5 inference failed")

    # ========================================================================
    # PARSING - Extrait les prédictions de la sortie YOLOv5
    # ========================================================================
    records: List[PredictionRecord] = []
    for line in output.splitlines():
        # Essaie de parser chaque ligne
        record = parse_prediction_line(line)
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError("Aucune ligne de prédiction exploitable n'a été trouvée dans la sortie YOLOv5.")

    # ========================================================================
    # POST-TRAITEMENT - Applique les seuils de confiance et normalise
    # ========================================================================
    for rec in records:
        # Trie les scores par confiance décroissante
        scores = sorted(rec.class_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_score = scores[0][1] if scores else rec.top1_score

        # Classifie selon les seuils définis (BDD >= 60%, INCERTITUDE 50-60%, HORS_BDD < 50%)
        if top1_score >= BDD_THRES:
            rec.status = "BDD"
        elif top1_score >= INCERTITUDE_THRES:
            rec.status = "INCERTITUDE"
        else:
            rec.status = "HORS_BDD"

        # Si hors dataset, renomme la classe en "autre" pour le rapport final
        if rec.status == "HORS_BDD":
            rec.top1_class = "autre"

    return records



# ============================================================================
# AFFICHAGE CONSOLE - Résultats pour une seule image
# ============================================================================

def print_single_image_result(record: PredictionRecord) -> None:
    """
    Affiche un résumé console propre pour l'analyse d'une seule image.
    Affiche la classe prédite, la confiance, le statut et tous les scores détaillés.
    
    Args:
        record: PredictionRecord contenant les données de la prédiction
    """
    print(console_text("\nAnalyse de l'image", Fore.CYAN, bright=True))
    print(f"Image : {record.image_path}")
    
    # Affiche "autre" si classe hors dataset, sinon la classe top-1
    display_class = "autre" if record.status == "HORS_BDD" else record.top1_class
    print(f"Classe top-1 : {console_text(display_class, Fore.GREEN, bright=True)}")
    print(f"Confiance : {record.top1_score * 100:.2f} %")
    print(f"Statut : {record.status if record.status != 'HORS_BDD' else 'AUTRE'}")
    
    # Affiche les probabilités pour toutes les classes, triées par score décroissant
    print("Probabilités détaillées :")
    for class_name, score in sorted(record.class_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {class_name:<20} {score * 100:6.2f} %")
    # Si un index vectoriel est disponible, interroger et afficher le voisin le plus proche
    try:
        from vector_index import query_image
        if VECTOR_DIR.exists() and (VECTOR_DIR / "index.faiss").exists():
            results = query_image(record.image_path, VECTOR_DIR, k=3)
            if results:
                print(console_text("\nVoisins visuels (index vectoriel):", Fore.MAGENTA, bright=True))
                for path, score in results:
                    print(f"  - {path}  score={score:.3f}")
    except Exception:
        # Ne pas planter l'affichage si l'index n'est pas disponible
        pass

# ============================================================================
# VISUALISATION GRAPHIQUE - Graphique de confiance pour une image
# ============================================================================

def plot_single_image(record: PredictionRecord, output_dir: Path) -> Path:
    """
    Crée un graphique en barres horizontales montrant la confiance par classe pour une image.
    
    Args:
        record: PredictionRecord avec les scores de confiance
        output_dir: Répertoire où sauvegarder le PNG
    
    Returns:
        Chemin vers le fichier PNG généré
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prépare les données: trie par score décroissant
    scores = sorted(record.class_scores.items(), key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in scores]
    values = [value * 100 for _, value in scores]  # Convertit en pourcentages
    
    # Palette de couleurs harmonieuse (max 5 classes différentes)
    colors = ["#2F80ED", "#27AE60", "#F2994A", "#9B51E0", "#56CCF2"][: len(values)]

    # Configure le style et la figure matplotlib
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.8 * len(labels) + 2)), dpi=160)
    
    # Crée le graphique en barres horizontales
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Classe top-1 en haut
    ax.set_xlabel("Confiance (%)")
    ax.set_title(f"Analyse: {record.image_path.name}", pad=12, weight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Ajoute les valeurs numériques au bout de chaque barre
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
            weight="bold",
        )

    # Sauvegarde la figure
    fig.tight_layout()
    output_path = output_dir / f"analyse_image_{record.image_path.stem}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output_path

# ============================================================================
# VISUALISATION GRAPHIQUE - Résumé statistique du dossier
# ============================================================================

def plot_folder_summary(
    total: int,
    correct: int,
    failed: int,
    status_counts: Counter[str],
    output_dir: Path,
) -> Path:
    """
    Crée un graphique professionnel avec le résumé complet du dossier analysé.
    Génère 2 figures: répartition BDD/INCERTITUDE/autre, et réussite/échec.
    
    Args:
        total: Nombre total d'images analysées
        correct: Nombre de prédictions correctes
        failed: Nombre d'erreurs
        status_counts: Compteur des statuts (BDD, INCERTITUDE, HORS_BDD)
        output_dir: Répertoire de sauvegarde
    
    Returns:
        Chemin vers le graphique principal PNG
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcule les pourcentages
    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0
    bdd_pct = (status_counts.get("BDD", 0) / total * 100.0) if total else 0.0
    incertitude_pct = (status_counts.get("INCERTITUDE", 0) / total * 100.0) if total else 0.0
    hors_bdd_count = status_counts.get("HORS_BDD", status_counts.get("autre", 0))
    hors_bdd_pct = (hors_bdd_count / total * 100.0) if total else 0.0

    # ====== FIGURE 1: Bilan global + Pie chart répartition =====
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 8), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax_top = fig.add_subplot(gs[0, :])
    ax_pie = fig.add_subplot(gs[1, 0])

    # Graphique haut: nombre total d'images
    ax_top.bar(["Images analysées"], [total], color="#2B7A78", edgecolor="white")
    ax_top.set_ylabel("Nombre d'images")
    ax_top.set_title("Bilan global de l'analyse du dossier", pad=12, weight="bold")
    ax_top.set_ylim(0, max(1, total * 1.15))
    ax_top.annotate(f"{total}", xy=(0, total), xytext=(0, 8), textcoords="offset points", ha="center", weight="bold")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Graphique bas: pie chart avec répartition BDD/INCERTITUDE/autre
    statuses = [status_counts.get(k, 0) for k in ("BDD", "INCERTITUDE", "autre")]
    labels = [f"BDD ({bdd_pct:.1f}%)", f"INCERTITUDE ({incertitude_pct:.1f}%)", f"autre ({hors_bdd_pct:.1f}%)"]
    colors_pie = ["#4C78A8", "#F2C14E", "#D95D39"]
    wedges, texts = ax_pie.pie(statuses, labels=labels, colors=colors_pie, startangle=90, wedgeprops={"width": 0.45, "edgecolor": "white"})
    ax_pie.set_title("Répartition décision modèle", weight="bold")

    fig.tight_layout()
    output_path = output_dir / "bilan_analyse_dossier.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # ====== FIGURE 2: Réussite vs Échec (barre stacked) =====
    fig2, ax2 = plt.subplots(figsize=(12, 3.5), dpi=160)
    bar_height = 0.8
    # Barre vert pour réussite + rouge pour échec
    ax2.barh([0], [success_pct], height=bar_height, color="#2ECC71", edgecolor="white")
    ax2.barh([0], [failure_pct], left=[success_pct], height=bar_height, color="#E74C3C", edgecolor="white")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Pourcentage")
    ax2.set_yticks([])
    ax2.set_title("Réussite vs Échec", weight="bold")
    # Ajoute les pourcentages sur les barres
    ax2.text(success_pct / 2 if success_pct else 2, 0, f"{success_pct:.1f}%", ha="center", va="center", weight="bold", fontsize=13)
    ax2.text(success_pct + failure_pct / 2 if failure_pct else max(success_pct + 2, 2), 0, f"{failure_pct:.1f}%", ha="center", va="center", weight="bold", color="white", fontsize=13)
    fig2.tight_layout()
    output_path2 = output_dir / "bilan_reussite_echec.png"
    fig2.savefig(output_path2, bbox_inches="tight")
    plt.show()
    plt.close(fig2)

    return output_path


# ============================================================================
# SAUVEGARDE JSON - Résumé des résultats au format JSON
# ============================================================================

def write_folder_summary_json(
    total: int,
    correct: int,
    failed: int,
    status_counts: Counter[str],
    confusion_counts: Counter[tuple[str, str]],
    output_dir: Path,
) -> Path:
    """
    Sauvegarde un résumé complet de l'analyse en fichier JSON.
    Inclut les statistiques globales et la matrice de confusion.
    
    Args:
        total: Nombre total d'images
        correct: Nombre de prédictions correctes
        failed: Nombre d'erreurs
        status_counts: Compteur des statuts
        confusion_counts: Compteur des confusions (true_label, pred_label) -> count
        output_dir: Répertoire de sauvegarde
    
    Returns:
        Chemin vers le fichier JSON généré
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extrait les compteurs par statut
    bdd = status_counts.get("BDD", 0)
    inc = status_counts.get("INCERTITUDE", 0)
    autre = status_counts.get("autre", status_counts.get("HORS_BDD", 0))

    # Calcule les pourcentages
    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0

    # Construit le dictionnaire de données
    data = {
        "total": total,
        "correct": correct,
        "failed": failed,
        "success_pct": round(success_pct, 2),
        "failure_pct": round(failure_pct, 2),
        "breakdown": {"BDD": bdd, "INCERTITUDE": inc, "autre": autre},
        # Liste les N plus grandes confusions (true_label -> pred_label)
        "confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in confusion_counts.most_common()
        ],
    }

    # Sauvegarde le JSON avec indentation et support UTF-8
    out = output_dir / "bilan_analyse_dossier.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

# ============================================================================
# AFFICHAGE CONSOLE - Résumé et confusion du dossier
# ============================================================================

def print_folder_summary(total: int, correct: int, failed: int, confusion_counts: Counter[tuple[str, str]]) -> None:
    """
    Affiche les statistiques du dossier et les 10 confusions les plus fréquentes.
    
    Args:
        total: Nombre total d'images
        correct: Nombre de prédictions correctes
        failed: Nombre d'erreurs
        confusion_counts: Compteur des confusions (true_label, pred_label) -> count
    """
    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0

    print(console_text("\nBilan du dossier", Fore.CYAN, bright=True))
    print(f"Nombre total d'images analysées : {total}")
    print(f"Nombre d'images correctement trouvées : {correct}")
    print(f"Nombre d'échecs : {failed}")
    print(f"Pourcentage global de réussite : {success_pct:.2f} %")
    print(f"Pourcentage global d'échec : {failure_pct:.2f} %")

    # Affiche les 10 confusions les plus fréquentes
    if confusion_counts:
        print(console_text("\nClasses les plus souvent confondues", Fore.YELLOW, bright=True))
        for (true_label, predicted_label), count in confusion_counts.most_common(10):
            print(f"  - {true_label} -> {predicted_label} : {count}")
    else:
        print(console_text("\nAucune confusion détectée.", Fore.GREEN, bright=True))

# ============================================================================
# ANALYSE SIMPLE - Analyse d'une seule image
# ============================================================================

def analyze_single_image(image_path: Path) -> None:
    """
    Analyse une seule image: lance la prédiction, affiche les résultats, joue le son approprié et génère un graphique.
    
    Logique de lecture audio :
    - BDD : joue le son de l'oiseau détecté
    - INCERTITUDE : joue le signal d'alerte (canon.mp3)
    - HORS_BDD : pas de son
    
    Args:
        image_path: Chemin vers l'image à analyser
    
    Raises:
        FileNotFoundError: Si l'image n'existe pas
    """
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    # Lance la prédiction YOLOv5
    records = run_yolov5_prediction(image_path)
    record = records[0]
    
    # Affiche les résultats en console
    print_single_image_result(record)

    # Sauvegarde l'image dans le dossier correspondant au statut détecté
    save_analyzed_image(image_path, record)
    
    # Lance la lecture audio appropriée selon le statut
    print()  # Ligne vide pour lisibilité
    if record.status == "BDD":
        # BDD : joue le son de l'oiseau
        play_bird_audio(record.top1_class)
    elif record.status == "INCERTITUDE":
        # INCERTITUDE : joue le signal d'alerte (canon)
        print(console_text("[AUDIO] Statut INCERTITUDE détecté - lecture du signal d'alerte...", Fore.YELLOW, bright=True))
        play_audio(AUDIO_ALERT)
    else:  # HORS_BDD
        # HORS_BDD : pas de son, message informatif
        print(console_text("[AUDIO] Impossible de jouer un son: oiseau non reconnu (AUTRE).", Fore.YELLOW))
    
    # Génère un graphique de confiance
    print()  # Ligne vide pour lisibilité
    chart_path = plot_single_image(record, RESULTS_DIR)
    print(console_text(f"Graphique enregistré dans: {chart_path}", Fore.GREEN, bright=True))
    
    # Avertit si la classe est hors dataset
    if record.status == "HORS_BDD":
        print(console_text("Attention: cet oiseau ne semble pas appartenir aux classes du dataset (AUTRE).", Fore.YELLOW, bright=True))

# ============================================================================
# ANALYSE DOSSIER - Analyse récursive d'un dossier d'images
# ============================================================================

def analyze_folder(folder_path: Path) -> None:
    """
    Analyse toutes les images d'un dossier (récursivement).
    Compare les prédictions aux étiquettes (noms des sous-dossiers).
    Une prédiction en INCERTITUDE est comptée comme une réussite pour le bilan.
    Génère des graphiques et un fichier JSON avec les résultats détaillés.
    
    Args:
        folder_path: Chemin vers le dossier contenant des images ou sous-dossiers d'images
    
    Raises:
        NotADirectoryError: Si le chemin n'est pas un dossier
        ValueError: Si aucune image trouvée
        RuntimeError: Si le nombre de prédictions ne correspond pas
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise NotADirectoryError(f"Dossier introuvable: {folder_path}")

    # Trouve toutes les images du dossier
    image_files = find_image_files(folder_path)
    if not image_files:
        raise ValueError(f"Aucune image compatible n'a été trouvée dans {folder_path}")

    # Initialise les compteurs
    correct = 0
    failed = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()

    print(console_text(f"\nAnalyse du dossier: {folder_path}", Fore.CYAN, bright=True))
    print(f"Nombre d'images détectées : {len(image_files)}")

    # ========================================================================
    # INFÉRENCE - Lance les prédictions YOLOv5 en mode batch
    # ========================================================================
    with tempfile.TemporaryDirectory(prefix="analyse_oiseaux_") as temp_dir:
        temp_path = Path(temp_dir)
        # Crée un fichier liste des images pour YOLOv5
        source_list = temp_path / "images_recursives.txt"
        source_list.write_text("\n".join(str(image_path) for image_path in image_files), encoding="utf-8")
        
        # Lance YOLOv5 avec la liste (mode batch, pas de sauvegarde)
        records = run_yolov5_prediction(source_list, save_outputs=False)

        # Vérifie que le nombre de prédictions correspond
        if len(records) != len(image_files):
            raise RuntimeError(
                f"Le nombre de prédictions ({len(records)}) ne correspond pas au nombre d'images détectées ({len(image_files)})."
            )

        # ====================================================================
        # ÉVALUATION - Compare prédictions vs étiquettes (nom des dossiers)
        # ====================================================================
        missing_images: List[Path] = []
        
        # Itère sur chaque image avec sa prédiction (avec barre de progression)
        for image_path, record in tqdm(list(zip(image_files, records)), desc="Analyse des images", unit="image"):
            # Incrémente le compteur du statut
            status_counts[record.status] += 1

            # Sauvegarde l'image dans le dossier correspondant au statut détecté
            save_analyzed_image(image_path, record)
            
            # Extrait l'étiquette vraie du nom du dossier parent
            true_label = normalize_label(image_path.parent.name)
            
            # Si hors dataset, traite comme 'autre' pour les statistiques
            if record.status == "HORS_BDD":
                predicted_label = "autre"
                missing_images.append(image_path)
            else:
                predicted_label = normalize_label(record.top1_class)

            # Une incertitude est considérée comme une réussite, même si la classe n'est pas identique.
            if record.status == "INCERTITUDE" or true_label == predicted_label:
                correct += 1
            else:
                failed += 1
                # Enregistre la confusion pour la matrice
                confusion_counts[(true_label, predicted_label)] += 1

    # ========================================================================
    # AFFICHAGE ET SAUVEGARDE - Résumés et graphiques
    # ========================================================================
    total = len(image_files)
    print_folder_summary(total, correct, failed, confusion_counts)

    # Normalise les clés de status pour l'affichage (HORS_BDD -> autre)
    display_status_counts: Counter[str] = Counter()
    for k, v in status_counts.items():
        if k == "HORS_BDD":
            display_status_counts["autre"] += v
        else:
            display_status_counts[k] += v

    # Génère les graphiques de résumé
    chart_path = plot_folder_summary(total, correct, failed, display_status_counts, RESULTS_DIR)
    print(console_text(f"Graphique sauvegardé dans: {chart_path}", Fore.GREEN, bright=True))
    
    # Sauvegarde le résumé JSON
    json_path = write_folder_summary_json(total, correct, failed, status_counts, confusion_counts, RESULTS_DIR)
    print(console_text(f"Résumé JSON sauvegardé dans: {json_path}", Fore.GREEN, bright=True))

    # Affiche les images potentiellement hors dataset
    if missing_images:
        print(console_text("\nImages potentiellement hors-dataset (AUTRE) :", Fore.YELLOW, bright=True))
        for p in missing_images:
            print(f" - {p}")


def analyze_camera_stream(camera_index: int = 0) -> None:
    """
    Lance une boucle de capture depuis la camera et analyse les captures à la demande.

    Args:
        camera_index: Index de la camera à ouvrir
    """
    print(console_text("\nMode camera", Fore.CYAN, bright=True))
    print("Touches disponibles :")
    print("  - c ou espace : capturer et analyser la frame courante")
    print("  - q ou Echap  : quitter")

    camera = open_camera(camera_index)
    temp_dir = Path(tempfile.mkdtemp(prefix="analyse_oiseaux_camera_"))

    # Event thread-safe pour indiquer si une analyse est en cours
    analysis_event = threading.Event()

    # Paramètres d'analyse vectorielle continue
    vector_enabled = VECTOR_DIR.exists() and (VECTOR_DIR / "index.faiss").exists()
    vector_interval = 2.0  # secondes entre deux analyses vectorielles
    vector_k = 3  # voisins retournés
    last_vector_time = 0.0
    clip_model = None
    clip_preprocess = None
    faiss_index = None
    faiss_mapping = None

    if vector_enabled:
        try:
            # Charge les dépendances et le modèle CLIP/FAISS une seule fois
            ensure_dependency("clip", "git+https://github.com/openai/CLIP.git")
            ensure_dependency("faiss", "faiss-cpu")
            ensure_dependency("Pillow")
            ensure_dependency("torch")
            import clip
            import torch
            import numpy as np
            from PIL import Image
            from vector_index import load_index

            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
            faiss_index, faiss_mapping = load_index(VECTOR_DIR)
            print(console_text(f"[VECTOR] Index chargé ({len(faiss_mapping)} items). Analyse continue activée.", Fore.MAGENTA, bright=True))
        except Exception as exc:
            print(console_text(f"[VECTOR] Impossible d'activer l'analyse vectorielle continue: {exc}", Fore.YELLOW, bright=True))
            vector_enabled = False

    # Cooldown entre captures manuelles (secondes)
    capture_cooldown = 3.0
    last_capture_time = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError(f"Impossible de lire une frame depuis la camera {camera_index}.")

            # Overlay status: affiche si une analyse est en cours ou temps restant avant capture
            now = time.time()
            in_progress = analysis_event.is_set()
            cooldown_remaining = max(0.0, last_capture_time + capture_cooldown - now)
            if in_progress:
                status_text = "ANALYSE EN COURS..."
                color = (0, 200, 200)
            elif cooldown_remaining > 0.0:
                status_text = f"Prêt dans {int(math.ceil(cooldown_remaining))}s"
                color = (0, 200, 200)
            else:
                status_text = "Appuyez 'c' ou Espace pour capturer"
                color = (200, 200, 200)

            cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.imshow("Analyse oiseaux - camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            # Capture on-demand
            if key in (ord("c"), ord(" ")):
                # Evite de lancer plusieurs analyses simultanées
                if analysis_event.is_set():
                    print(console_text("[CAMERA] Analyse en cours, veuillez patienter...", Fore.YELLOW, bright=True))
                else:
                    frame_path = capture_frame_to_tempfile(frame, temp_dir)
                    print(console_text(f"[CAMERA] Frame capturée: {frame_path.name}", Fore.CYAN, bright=True))

                    # Lance l'analyse dans un thread séparé pour ne pas bloquer l'UI
                    def _analyze_async(path: Path):
                        try:
                            analysis_event.set()
                            analyze_single_image(path)
                        except Exception as e:
                            print(console_text(f"[CAMERA] Erreur durant l'analyse asynchrone: {e}", Fore.RED, bright=True))
                        finally:
                            try:
                                path.unlink()
                            except Exception:
                                pass
                            analysis_event.clear()

                    # marque le début du cooldown et lance le thread
                    last_capture_time = time.time()
                    t = threading.Thread(target=_analyze_async, args=(frame_path,), daemon=True)
                    t.start()

            # Analyse vectorielle continue (périodique)
            try:
                if vector_enabled and (time.time() - last_vector_time) >= vector_interval and clip_model is not None:
                    # Prépare l'image PIL pour CLIP
                    import numpy as _np
                    from PIL import Image as _Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = _Image.fromarray(img_rgb)
                    inp = clip_preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = clip_model.encode_image(inp)
                    emb_np = emb.cpu().numpy().astype('float32')
                    emb_np = emb_np / (_np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10)
                    D, I = faiss_index.search(emb_np, vector_k)
                    # Affiche les voisins sur la frame
                    y0 = 20
                    for rank, (score, idx) in enumerate(zip(D[0], I[0])):
                        if idx < 0:
                            continue
                        mapped = faiss_mapping.get(str(int(idx))) or faiss_mapping.get(int(idx)) or str(idx)
                        text = f"#{rank+1}: {Path(mapped).name} ({float(score):.3f})"
                        cv2.putText(frame, text, (10, y0 + rank * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                    last_vector_time = time.time()
            except Exception:
                # Ne pas interrompre la boucle camera pour une erreur vectorielle
                pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
        shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================================================
# INTERACTION UTILISATEUR - Menus et input
# ============================================================================

def ask_user_choice() -> str:
    """
    Affiche un menu et demande à l'utilisateur de choisir une option.
    
    Returns:
        "1" pour analyser une seule image, "2" pour analyser un dossier, "3" pour utiliser la camera
    """
    print(console_text("\n=== Analyse oiseaux YOLOv5 ===", Fore.MAGENTA, bright=True))
    print("1. Analyser une seule image")
    print("2. Analyser un dossier complet d'images")
    print("3. Analyser depuis la camera")
    choice = input("Votre choix (1/2/3) : ").strip()
    while choice not in {"1", "2", "3"}:
        print(console_text("Choix invalide. Réessaie avec 1, 2 ou 3.", Fore.RED, bright=True))
        choice = input("Votre choix (1/2/3) : ").strip()
    return choice


def ask_camera_index() -> int:
    """
    Demande à l'utilisateur l'index de la caméra à ouvrir.

    Returns:
        Index caméra saisi, ou 0 si la saisie est vide/invalide
    """
    raw_value = input("Index de la camera (défaut 0) : ").strip()
    if not raw_value:
        return 0
    try:
        return int(raw_value)
    except ValueError:
        print(console_text("Index invalide, utilisation de la camera 0.", Fore.YELLOW, bright=True))
        return 0


def ask_path(prompt: str) -> Path:
    """
    Demande un chemin à l'utilisateur et le retourne en Path.
    
    Args:
        prompt: Texte affiché pour demander le chemin
    
    Returns:
        Path parsed du chemin saisi
    """
    raw_path = input(prompt).strip().strip('"')
    return Path(raw_path)

# ============================================================================
# INITIALISATION - Vérifications du projet
# ============================================================================

def ensure_project_environment() -> None:
    """
    Vérifie que l'environnement du projet est correct:
    - Crée le répertoire de résultats
    - Vérifie que les poids du modèle existent
    - Vérifie que le script YOLOv5 existe
    
    Raises:
        FileNotFoundError: Si les poids ou le script manquent
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ENREGISTREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Poids introuvables: {WEIGHTS}")
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError(f"Script YOLOv5 introuvable: {PREDICT_SCRIPT}")

# ============================================================================
# POINT D'ENTRÉE - Main function
# ============================================================================

def main() -> None:
    """
    Point d'entrée principal du script. Gère le flux d'exécution:
    1. Vérifie l'environnement
    2. Demande à l'utilisateur son choix (image ou dossier)
    3. Lance l'analyse appropriée
    4. Gère les erreurs et affiche les messages
    """
    try:
        # Vérifie que tout est en place
        ensure_project_environment()
        
        # Demande le choix de l'utilisateur
        choice = ask_user_choice()

        if choice == "1":
            # Analyse d'une seule image
            image_path = ask_path("Chemin complet de l'image : ")
            analyze_single_image(image_path)
        elif choice == "2":
            # Analyse d'un dossier
            folder_path = ask_path("Chemin complet du dossier : ")
            analyze_folder(folder_path)
        else:
            # Analyse depuis la camera
            camera_index = ask_camera_index()
            analyze_camera_stream(camera_index)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        # Erreurs de fichier/dossier
        print(console_text(f"[ERREUR] {exc}", Fore.RED, bright=True))
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        # Erreur lors de l'installation automatique
        print(console_text("[ERREUR] L'installation automatique d'une dépendance a échoué.", Fore.RED, bright=True))
        print(exc)
        sys.exit(1)
    except RuntimeError as exc:
        # Erreurs d'exécution (YOLOv5, prédiction, etc.)
        print(console_text(f"[ERREUR] {exc}", Fore.RED, bright=True))
        sys.exit(1)

# ============================================================================
# POINT D'EXÉCUTION - Script principal
# ============================================================================

if __name__ == "__main__":
    main()
