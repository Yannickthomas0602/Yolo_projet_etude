from pathlib import Path
import argparse
import random
import shutil
from typing import List


def gather_images(src: Path, exts=None) -> List[Path]:
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = [p for p in src.rglob('*') if p.suffix.lower() in exts and p.is_file()]
    return files


def make_dirs(base: Path):
    for part in ('train', 'validation', 'test'):
        (base / part).mkdir(parents=True, exist_ok=True)


def split_and_copy(files: List[Path], dest_base: Path, cls_name: str, ratios=(0.7, 0.2, 0.1), seed=42, move=False):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    mapping = {
        'train': train_files,
        'validation': val_files,
        'test': test_files,
    }

    for part, part_files in mapping.items():
        dst_dir = dest_base / part / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in part_files:
            dst = dst_dir / src.name
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

    return {k: len(v) for k, v in mapping.items()}


def main():
    parser = argparse.ArgumentParser(description='Prépare la classe "autre" dans dataset_oiseaux')
    parser.add_argument('--source', '-s', required=True, help='Dossier contenant les images à ajouter (récursif)')
    parser.add_argument('--dataset', '-d', default=str(Path(__file__).resolve().parents[1] / 'dataset_oiseaux'), help='Chemin vers dataset_oiseaux')
    parser.add_argument('--class-name', '-c', default='autre', help='Nom de la classe à créer (par défaut: autre)')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--move', action='store_true', help='Déplacer les fichiers au lieu de copier')

    args = parser.parse_args()
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        print(f"Source introuvable: {src}")
        return

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        print(f"Dataset introuvable: {dataset}")
        return

    make_dirs(dataset)
    files = gather_images(src)
    if not files:
        print(f"Aucune image trouvée dans {src}")
        return

    counts = split_and_copy(files, dataset, args.class_name, ratios=(args.train_ratio, args.val_ratio, args.test_ratio), seed=args.seed, move=args.move)

    print('Import terminé')
    print(f"Classe: {args.class_name}")
    print(f"Images trouvées: {len(files)}")
    print(f"Répartition: train={counts['train']}, validation={counts['validation']}, test={counts['test']}")
    print('Vérifie ensuite que les splits sont corrects et lance l\'entraînement.')


if __name__ == '__main__':
    main()
