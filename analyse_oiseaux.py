from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import argparse
import tempfile


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "train-cls" / "exp_retrain" / "weights" / "best.pt"
PREDICT_SCRIPT = ROOT / "classify" / "predict.py"
RESULTS_DIR = ROOT / "results"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
# Post-processing hyperparameters (no retrain required)
# Status is based ONLY on top-1 confidence:
# - BDD if top1 >= BDD_THRES
# - INCERTITUDE if INCERTITUDE_THRES <= top1 < BDD_THRES
# - HORS_BDD if top1 < INCERTITUDE_THRES
BDD_THRES = 0.60
INCERTITUDE_THRES = 0.50


def ensure_dependency(module_name: str, pip_name: str | None = None) -> None:
    """Install a missing dependency in the current virtual environment."""
    if importlib.util.find_spec(module_name) is not None:
        return

    package_name = pip_name or module_name
    print(f"[INFO] Module manquant détecté: {module_name}. Installation en cours avec pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def ensure_runtime_dependencies() -> None:
    """Ensure the plotting and progress libraries are available before importing them."""
    for module_name in ("matplotlib", "tqdm", "colorama"):
        ensure_dependency(module_name)


ensure_runtime_dependencies()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from tqdm import tqdm  # noqa: E402

try:
    from colorama import Fore, Style, init as colorama_init  # noqa: E402

    colorama_init(autoreset=True)
except Exception:  # pragma: no cover - fallback if colorama import fails unexpectedly
    class _FallbackColor:
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
        BRIGHT = ""
        NORMAL = ""
        RESET_ALL = ""

    Fore = _FallbackColor()
    Style = _FallbackStyle()


@dataclass
class PredictionRecord:
    """Prediction extracted from YOLOv5 classification output."""

    image_path: Path
    top1_class: str
    top1_score: float
    status: str
    class_scores: Dict[str, float]
    raw_line: str


PREDICTION_LINE_RE = re.compile(
    r"^image\s+\d+/\d+\s+(?P<path>.+):\s+"
    r"(?P<size>\d+x\d+)\s+"
    r"\[(?P<status>BDD|INCERTITUDE|HORS_BDD)\]\s+"
    r"(?P<top1_class>.+?)\s+(?P<top1_score>[0-9]*\.?[0-9]+),\s+"
    r"(?P<rest>.+),\s+(?P<time>[0-9]*\.?[0-9]+)ms$"
)


def console_text(text: str, color: str = "", bright: bool = False) -> str:
    prefix = f"{Style.BRIGHT if bright else ''}{color}"
    return f"{prefix}{text}{Style.RESET_ALL}"


def normalize_label(value: str) -> str:
    """Normalize labels for robust folder-vs-prediction comparisons."""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = normalized.lower().strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def find_image_files(folder: Path) -> List[Path]:
    """Return supported images under a folder, recursively sorted by path."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_prediction_line(line: str) -> PredictionRecord | None:
    """Parse one YOLOv5 classification log line into a structured record."""
    match = PREDICTION_LINE_RE.match(line.strip())
    if match is None:
        return None

    image_path = Path(match.group("path"))
    top1_class = match.group("top1_class").strip()
    top1_score = float(match.group("top1_score"))
    status = match.group("status")

    class_scores: Dict[str, float] = {top1_class: top1_score}
    rest = match.group("rest").strip()
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
    """Build a short unique name for YOLOv5 output folders."""
    base_name = normalize_label(source.stem if source.is_file() else source.name) or "analysis"
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{base_name}_{digest}"


def run_yolov5_prediction(source: Path, save_outputs: bool = True) -> List[PredictionRecord]:
    """Launch the official YOLOv5 classifier in a subprocess and capture its predictions."""
    command = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--weights",
        str(WEIGHTS),
        "--source",
        str(source),
        "--device",
        "0",
        "--bdd-thres",
        "0.60",
        "--uncertainty-thres",
        "0.30",
    ]

    if save_outputs:
        command.extend(
            [
                "--save-txt",
                "--project",
                str(RESULTS_DIR),
                "--name",
                build_run_name(source),
                "--exist-ok",
            ]
        )
    else:
        command.append("--nosave")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if completed.returncode != 0:
        print(console_text("[ERREUR] La commande YOLOv5 a échoué.", Fore.RED, bright=True))
        print(output)
        raise RuntimeError("YOLOv5 inference failed")

    records: List[PredictionRecord] = []
    for line in output.splitlines():
        record = parse_prediction_line(line)
        if record is not None:
            records.append(record)

    if not records:
        raise RuntimeError("Aucune ligne de prédiction exploitable n'a été trouvée dans la sortie YOLOv5.")

    # Post-process predictions using only top-1 confidence
    for rec in records:
        # Normalize keys for safety and use top-1 only for status.
        scores = sorted(rec.class_scores.items(), key=lambda kv: kv[1], reverse=True)
        top1_score = scores[0][1] if scores else rec.top1_score

        if top1_score >= BDD_THRES:
            rec.status = "BDD"
        elif top1_score >= INCERTITUDE_THRES:
            rec.status = "INCERTITUDE"
        else:
            rec.status = "HORS_BDD"

        # ensure top1_class becomes 'autre' when status is HORS_BDD for downstream reporting
        if rec.status == "HORS_BDD":
            rec.top1_class = "autre"

    return records


def print_single_image_result(record: PredictionRecord) -> None:
    """Display a clean console summary for a single image analysis."""
    print(console_text("\nAnalyse de l'image", Fore.CYAN, bright=True))
    print(f"Image : {record.image_path}")
    display_class = "autre" if record.status == "HORS_BDD" else record.top1_class
    print(f"Classe top-1 : {console_text(display_class, Fore.GREEN, bright=True)}")
    print(f"Confiance : {record.top1_score * 100:.2f} %")
    print(f"Statut : {record.status if record.status != 'HORS_BDD' else 'AUTRE'}")
    print("Probabilités détaillées :")
    for class_name, score in sorted(record.class_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {class_name:<20} {score * 100:6.2f} %")


def plot_single_image(record: PredictionRecord, output_dir: Path) -> Path:
    """Create a polished bar chart for one image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = sorted(record.class_scores.items(), key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in scores]
    values = [value * 100 for _, value in scores]
    colors = ["#2F80ED", "#27AE60", "#F2994A", "#9B51E0", "#56CCF2"][: len(values)]

    # Improved single-image chart: horizontal bars, clear labels, soft palette
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.8 * len(labels) + 2)), dpi=160)
    y_pos = list(range(len(labels)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Confiance (%)")
    ax.set_title(f"Analyse: {record.image_path.name}", pad=12, weight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # annotate values at end of bars
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
            weight="bold",
        )

    fig.tight_layout()
    output_path = output_dir / f"analyse_image_{record.image_path.stem}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return output_path


def plot_folder_summary(
    total: int,
    correct: int,
    failed: int,
    status_counts: Counter[str],
    output_dir: Path,
) -> Path:
    """Create a professional summary chart for folder analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0
    bdd_pct = (status_counts.get("BDD", 0) / total * 100.0) if total else 0.0
    incertitude_pct = (status_counts.get("INCERTITUDE", 0) / total * 100.0) if total else 0.0
    hors_bdd_count = status_counts.get("HORS_BDD", status_counts.get("autre", 0))
    hors_bdd_pct = (hors_bdd_count / total * 100.0) if total else 0.0

    # Improved folder chart: top -> total images, bottom -> pie for statuses + stacked bar for success/fail
    plt.style.use("seaborn-v0_8-whitegrid")
    # Main summary: total + pie/donut
    fig = plt.figure(figsize=(12, 8), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

    ax_top = fig.add_subplot(gs[0, :])
    ax_pie = fig.add_subplot(gs[1, 0])

    # Top: total
    ax_top.bar(["Images analysées"], [total], color="#2B7A78", edgecolor="white")
    ax_top.set_ylabel("Nombre d'images")
    ax_top.set_title("Bilan global de l'analyse du dossier", pad=12, weight="bold")
    ax_top.set_ylim(0, max(1, total * 1.15))
    ax_top.annotate(f"{total}", xy=(0, total), xytext=(0, 8), textcoords="offset points", ha="center", weight="bold")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Pie/donut for BDD / INCERTITUDE / HORS_BDD
    # support either key 'HORS_BDD' or the renamed 'autre'
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

    # Separate, larger figure for success / failure to improve readability
    fig2, ax2 = plt.subplots(figsize=(12, 3.5), dpi=160)
    bar_height = 0.8
    ax2.barh([0], [success_pct], height=bar_height, color="#2ECC71", edgecolor="white")
    ax2.barh([0], [failure_pct], left=[success_pct], height=bar_height, color="#E74C3C", edgecolor="white")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Pourcentage")
    ax2.set_yticks([])
    ax2.set_title("Réussite vs Échec", weight="bold")
    ax2.text(success_pct / 2 if success_pct else 2, 0, f"{success_pct:.1f}%", ha="center", va="center", weight="bold", fontsize=13)
    ax2.text(success_pct + failure_pct / 2 if failure_pct else max(success_pct + 2, 2), 0, f"{failure_pct:.1f}%", ha="center", va="center", weight="bold", color="white", fontsize=13)
    fig2.tight_layout()
    output_path2 = output_dir / "bilan_reussite_echec.png"
    fig2.savefig(output_path2, bbox_inches="tight")
    plt.show()
    plt.close(fig2)

    return output_path


def write_folder_summary_json(
    total: int,
    correct: int,
    failed: int,
    status_counts: Counter[str],
    confusion_counts: Counter[tuple[str, str]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bdd = status_counts.get("BDD", 0)
    inc = status_counts.get("INCERTITUDE", 0)
    autre = status_counts.get("autre", status_counts.get("HORS_BDD", 0))

    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0

    data = {
        "total": total,
        "correct": correct,
        "failed": failed,
        "success_pct": round(success_pct, 2),
        "failure_pct": round(failure_pct, 2),
        "breakdown": {"BDD": bdd, "INCERTITUDE": inc, "autre": autre},
        "confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in confusion_counts.most_common()
        ],
    }

    out = output_dir / "bilan_analyse_dossier.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def print_folder_summary(total: int, correct: int, failed: int, confusion_counts: Counter[tuple[str, str]]) -> None:
    """Display the folder-level metrics and the most common confusions."""
    success_pct = (correct / total * 100.0) if total else 0.0
    failure_pct = (failed / total * 100.0) if total else 0.0

    print(console_text("\nBilan du dossier", Fore.CYAN, bright=True))
    print(f"Nombre total d'images analysées : {total}")
    print(f"Nombre d'images correctement trouvées : {correct}")
    print(f"Nombre d'échecs : {failed}")
    print(f"Pourcentage global de réussite : {success_pct:.2f} %")
    print(f"Pourcentage global d'échec : {failure_pct:.2f} %")

    if confusion_counts:
        print(console_text("\nClasses les plus souvent confondues", Fore.YELLOW, bright=True))
        for (true_label, predicted_label), count in confusion_counts.most_common(10):
            print(f"  - {true_label} -> {predicted_label} : {count}")
    else:
        print(console_text("\nAucune confusion détectée.", Fore.GREEN, bright=True))


def analyze_single_image(image_path: Path) -> None:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image introuvable: {image_path}")

    records = run_yolov5_prediction(image_path)
    record = records[0]
    print_single_image_result(record)
    chart_path = plot_single_image(record, RESULTS_DIR)
    print(console_text(f"Graphique enregistré dans: {chart_path}", Fore.GREEN, bright=True))
    if record.status == "HORS_BDD":
        print(console_text("Attention: cet oiseau ne semble pas appartenir aux classes du dataset (AUTRE).", Fore.YELLOW, bright=True))


def analyze_folder(folder_path: Path) -> None:
    if not folder_path.exists() or not folder_path.is_dir():
        raise NotADirectoryError(f"Dossier introuvable: {folder_path}")

    image_files = find_image_files(folder_path)
    if not image_files:
        raise ValueError(f"Aucune image compatible n'a été trouvée dans {folder_path}")

    correct = 0
    failed = 0
    confusion_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()

    print(console_text(f"\nAnalyse du dossier: {folder_path}", Fore.CYAN, bright=True))
    print(f"Nombre d'images détectées : {len(image_files)}")

    with tempfile.TemporaryDirectory(prefix="analyse_oiseaux_") as temp_dir:
        temp_path = Path(temp_dir)
        source_list = temp_path / "images_recursives.txt"
        source_list.write_text("\n".join(str(image_path) for image_path in image_files), encoding="utf-8")
        records = run_yolov5_prediction(source_list, save_outputs=False)

        if len(records) != len(image_files):
            raise RuntimeError(
                f"Le nombre de prédictions ({len(records)}) ne correspond pas au nombre d'images détectées ({len(image_files)})."
            )

        missing_images: List[Path] = []
        for image_path, record in tqdm(list(zip(image_files, records)), desc="Analyse des images", unit="image"):
            status_counts[record.status] += 1
            true_label = normalize_label(image_path.parent.name)
            # If model marks HORS_BDD, treat as 'autre' category for reporting
            if record.status == "HORS_BDD":
                predicted_label = "autre"
                missing_images.append(image_path)
            else:
                predicted_label = normalize_label(record.top1_class)

            if true_label == predicted_label:
                correct += 1
            else:
                failed += 1
                confusion_counts[(true_label, predicted_label)] += 1

    total = len(image_files)
    print_folder_summary(total, correct, failed, confusion_counts)

    # Normalize status keys for display: map HORS_BDD -> 'autre'
    display_status_counts: Counter[str] = Counter()
    for k, v in status_counts.items():
        if k == "HORS_BDD":
            display_status_counts["autre"] += v
        else:
            display_status_counts[k] += v

    chart_path = plot_folder_summary(total, correct, failed, display_status_counts, RESULTS_DIR)
    print(console_text(f"Graphique sauvegardé dans: {chart_path}", Fore.GREEN, bright=True))
    json_path = write_folder_summary_json(total, correct, failed, status_counts, confusion_counts, RESULTS_DIR)
    print(console_text(f"Résumé JSON sauvegardé dans: {json_path}", Fore.GREEN, bright=True))

    if missing_images:
        print(console_text("\nImages potentiellement hors-dataset (AUTRE) :", Fore.YELLOW, bright=True))
        for p in missing_images:
            print(f" - {p}")


def ask_user_choice() -> str:
    print(console_text("\n=== Analyse oiseaux YOLOv5 ===", Fore.MAGENTA, bright=True))
    print("1. Analyser une seule image")
    print("2. Analyser un dossier complet d'images")
    choice = input("Votre choix (1/2) : ").strip()
    while choice not in {"1", "2"}:
        print(console_text("Choix invalide. Réessaie avec 1 ou 2.", Fore.RED, bright=True))
        choice = input("Votre choix (1/2) : ").strip()
    return choice


def ask_path(prompt: str) -> Path:
    raw_path = input(prompt).strip().strip('"')
    return Path(raw_path)


def ensure_project_environment() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Poids introuvables: {WEIGHTS}")
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError(f"Script YOLOv5 introuvable: {PREDICT_SCRIPT}")


def main() -> None:
    try:
        ensure_project_environment()
        choice = ask_user_choice()

        if choice == "1":
            image_path = ask_path("Chemin complet de l'image : ")
            analyze_single_image(image_path)
        else:
            folder_path = ask_path("Chemin complet du dossier : ")
            analyze_folder(folder_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(console_text(f"[ERREUR] {exc}", Fore.RED, bright=True))
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(console_text("[ERREUR] L'installation automatique d'une dépendance a échoué.", Fore.RED, bright=True))
        print(exc)
        sys.exit(1)
    except RuntimeError as exc:
        print(console_text(f"[ERREUR] {exc}", Fore.RED, bright=True))
        sys.exit(1)


if __name__ == "__main__":
    main()