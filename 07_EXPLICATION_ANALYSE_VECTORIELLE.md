# Analyse vectorielle expliquée — fonctionnement concret

Ce document explique, pas à pas et de manière concrète, comment fonctionne l'analyse vectorielle utilisée dans ce projet (CLIP pour les embeddings + FAISS pour l'indexation). Il montre ce qui se passe depuis l'image d'entrée jusqu'à la décision métier.

## 1 — Vue d'ensemble du pipeline

1. Prétraitement de l'image (resize, conversion RGB, normalisation).
2. Encodage par un modèle d'embeddings (ex: CLIP) → vecteur $\\mathbf{v} \\in \\mathbb{R}^d$.
3. Normalisation du vecteur (L2) pour obtenir $\\hat{\\mathbf{v}} = \\mathbf{v} / \\|\\mathbf{v}\\|$.
4. Recherche des K plus proches voisins dans un index FAISS (inner product sur vecteurs normalisés ⇒ cosinus).
5. Récupération des chemins/métadonnées correspondants via `mapping.json`.
6. Combinaison des scores de similarité avec la sortie du classifieur (YOLOv5) pour la décision finale.

## 2 — Pourquoi normaliser et utiliser l'inner product

Si les vecteurs sont normalisés (norme L2 = 1), le produit intérieur est égal au cosinus d'angle:

$$
\\cos(\\theta) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|\\,\\|\\mathbf{v}\\|}
$$

Avec $\\|\\mathbf{u}\\|=\\|\\mathbf{v}\\|=1$ on a $\\cos(\\theta)=\\mathbf{u}\\cdot\\mathbf{v}$. Les valeurs proches de 1 indiquent une forte similarité visuelle.

Avantages :
- Comparaison stable entre éléments.
- FAISS optimise l'inner product pour des recherches rapides (IndexFlatIP, ou indexes ANN plus avancés).

## 3 — Étapes concrètes côté code (exemple simplifié)

```python
from pathlib import Path
from vector_index import query_image, load_index
from clip import load as clip_load
import numpy as np

# Charger index (si existe)
index, mapping = load_index(Path('vectors'))

# Encoder l'image (ex: CLIP)
model, preprocess = clip_load('ViT-B/32', device='cpu')
img_t = preprocess(Image.open('test.jpg').convert('RGB')).unsqueeze(0)
with torch.no_grad():
    emb = model.encode_image(img_t).cpu().numpy()
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

# Rechercher les K voisins
D, I = index.search(emb.astype('float32'), k=5)  # D: scores, I: indices
results = [(mapping[str(int(idx))], float(score)) for score, idx in zip(D[0], I[0]) if idx >= 0]
```

## 4 — Combiner vectoriel + classifieur (règles pratiques)

- Cas A — Classifieur confiant (top1_score >= BDD_THRES) : accepter la prédiction.
- Cas B — Classifieur incertain (INCERTITUDE_THRES ≤ top1_score < BDD_THRES) :
  - si le voisin FAISS a `score >= SIM_THRESH_HIGH` (ex: 0.80) et la même classe → considérer comme probable ; sinon marquer `INCERTITUDE`.
- Cas C — Classifieur faible (top1_score < INCERTITUDE_THRES) :
  - si plusieurs voisins FAISS avec la même classe et `score >= SIM_THRESH_LOW` (ex: 0.30–0.50) → suggérer la classe aux réviseurs humains ; sinon `HORS_BDD`.

Exemples de valeurs recommandées (à calibrer) :
- `SIM_THRESH_HIGH = 0.80`
- `SIM_THRESH_LOW = 0.30`

## 5 — Interprétation des scores FAISS

- Score ≈ 1.0 : quasi-identité visuelle (même photo ou très proche).
- 0.6–0.8 : forte similarité (mêmes formes/texture/coeurs visuels).
- 0.3–0.6 : ressemblance faible → utile comme signal complémentaire mais non définitif.

Important : le seuil utile dépend du corpus. Sur des données bruyantes, augmenter `SIM_THRESH_HIGH` pour réduire faux positifs.

## 6 — Flux de construction d'index (hors-ligne)

1. Rassembler les images de référence (`dataset_oiseaux`, `enregistrements`).
2. Calculer embeddings CLIP pour chaque image et normaliser.
3. Construire l'index FAISS (`IndexFlatIP` pour débuter).
4. Sauvegarder `vectors/index.faiss` et `vectors/mapping.json`.

Commande d'exemple :
```bash
python vector_index.py --build --sources dataset_oiseaux enregistrements --out vectors
```

## 7 — Cas d'usage en mode caméra

- Ne pas calculer un embedding sur chaque frame (trop coûteux).
- Exemples de stratégie : calculer un embedding toutes les N secondes, ou quand le classifieur indique un changement de classe.
- Déclencher une capture persistante et indexation locale uniquement si la similarité dépasse un seuil calibré.

## 8 — Limitations et risques

- Biais d'index : si l'index contient des erreurs d'étiquetage, la similarité peut renforcer ces erreurs.
- Sensibilité aux conditions d'éclairage et de cadrage : embeddings peuvent varier fortement.
- Coût mémoire : FAISS stocke des vecteurs en mémoire ; pour très grands corpus, utiliser des indexes quantifiés ou des solutions distribuées.

## 9 — Conseils pratiques pour calibration

1. Construire une petite base de validation avec paires `image_query` + `label_correct`.
2. Mesurer distribution des scores FAISS intra-classe vs inter-classe.
3. Choisir `SIM_THRESH_LOW` et `SIM_THRESH_HIGH` en fonction des distributions (par ex. seuil à 95e percentile des scores inter-classe).

## 10 — Ressources rapides

- Code du projet : [yolov5/vector_index.py](vector_index.py)
- Documentation FAISS : https://github.com/facebookresearch/faiss
- CLIP (OpenAI) : https://github.com/openai/CLIP

---

Si tu veux, j'intègre ces règles directement dans `analyse_oiseaux.py` (fonctions utilitaires + seuils config) et je lance un petit script de calibration sur `enregistrements/`. Tu veux que je le fasse ?
