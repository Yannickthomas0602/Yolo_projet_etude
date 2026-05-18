"""
Construction et interrogation d'un index vectoriel d'images using CLIP + FAISS.

Usage:
  - Construire l'index (une seule fois):
      python vector_index.py --build --sources dataset_oiseaux enregistrements

  - Requêter (exemple):
      from vector_index import query_image
      query_image(Path('path/to/img.jpg'))

Le script tente d'installer automatiquement les dépendances (CLIP, faiss-cpu).
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

def ensure_dependency(module_name: str, pip_name: str | None = None) -> None:
    import importlib.util
    if importlib.util.find_spec(module_name) is not None:
        return
    package = pip_name or module_name
    print(f"[INFO] Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def build_index(sources: List[Path], output_dir: Path, batch_size: int = 32) -> None:
    """Build a FAISS index from images found in the given sources.

    Args:
        sources: list of folders to scan for images
        output_dir: where to save index and mapping
    """
    ensure_dependency("clip", "git+https://github.com/openai/CLIP.git")
    ensure_dependency("faiss", "faiss-cpu")
    ensure_dependency("Pillow")
    ensure_dependency("tqdm")

    import clip
    import faiss
    from PIL import Image
    import torch
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_paths: List[Path] = []
    for s in sources:
        if not s.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(sorted(s.rglob(ext)))

    if not image_paths:
        print("[WARN] Aucun fichier image trouvé dans les sources fournies.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    mapping = {}

    with torch.no_grad():
        for idx in tqdm(range(0, len(image_paths), batch_size), desc="Embedding images"):
            batch = image_paths[idx: idx + batch_size]
            imgs = [preprocess(Image.open(str(p)).convert("RGB")) for p in batch]
            imgs_t = torch.stack(imgs).to(device)
            emb = model.encode_image(imgs_t)
            emb = emb.cpu().numpy()
            # normalize
            norms = (emb ** 2).sum(axis=1, keepdims=True) ** 0.5
            emb = emb / (norms + 1e-10)
            for i, p in enumerate(batch):
                mapping[len(vectors)] = str(p)
                vectors.append(emb[i])

    import numpy as np

    xb = np.vstack(vectors).astype('float32')
    dim = xb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors -> cosine
    index.add(xb)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    (output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Index sauvegardé dans: {output_dir}")


def load_index(index_dir: Path):
    ensure_dependency("faiss", "faiss-cpu")
    ensure_dependency("clip", "git+https://github.com/openai/CLIP.git")
    import faiss
    import clip
    import torch

    index = faiss.read_index(str(index_dir / "index.faiss"))
    mapping = json.loads((index_dir / "mapping.json").read_text(encoding="utf-8"))
    return index, mapping


def query_image(image_path: Path, index_dir: Path, k: int = 5) -> List[Tuple[str, float]]:
    """Return list of (image_path, score) for top-k similar images (cosine-like score).

    Score is inner product on normalized CLIP vectors (1.0 best).
    """
    ensure_dependency("clip", "git+https://github.com/openai/CLIP.git")
    ensure_dependency("faiss", "faiss-cpu")
    ensure_dependency("Pillow")
    import clip
    import torch
    from PIL import Image
    import faiss
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    img = preprocess(Image.open(str(image_path)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
    emb = emb.cpu().numpy().astype('float32')
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    index, mapping = load_index(index_dir)
    D, I = index.search(emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append((mapping.get(str(int(idx))) or mapping.get(idx) or str(idx), float(score)))
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--sources", nargs="*", default=["dataset_oiseaux", "enregistrements"]) 
    p.add_argument("--out", default="vectors")
    args = p.parse_args()
    if args.build:
        srcs = [Path(s) for s in args.sources]
        build_index(srcs, Path(args.out))