
# Mode vectoriel : CLIP + FAISS

Ce document explique comment le projet utilise CLIP pour produire des embeddings et FAISS pour indexer et rechercher des images similaires.

## Principe

1. Construire un index hors ligne à partir d'un ensemble d'images de référence.
2. Pour chaque image d'entrée, calculer un embedding via CLIP.
3. Interroger l'index FAISS pour récupérer les voisins.

## 1) Prérequis

- Installer `faiss` (ou `faiss-cpu`) et `transformers`/`clip` selon ton environnement.
- Avoir un dossier `vectors/` où seront écrits `index.faiss` et `mapping.json`.

Exemples d'installation (Linux/Windows/GPU) :

```bash
pip install faiss-cpu
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

ou pour GPU (si supporté) :

```bash
pip install faiss-gpu
```

## 2) Construction de l'index

Le script `vector_index.py` sert à construire l'index. Le flux typique est :

1. Rassembler les images sources (par ex. `dataset_oiseaux`, `enregistrements`).
2. Pour chaque image, calculer l'embedding CLIP.
3. Normaliser chaque embedding (norme L2) pour utiliser le produit intérieur comme critère de similarité.
4. Insérer les vecteurs dans un index FAISS adapté (ex: `IndexFlatIP` pour inner product) et sauvegarder `index.faiss`.

Commandes recommandées :

```bash
python vector_index.py --build --sources dataset_oiseaux enregistrements --out vectors
```

## 3) Format de `mapping.json`

Le fichier `mapping.json` associe chaque identifiant de vecteur (int) au chemin de l'image correspondante ainsi qu'à des métadonnées utiles (classe, timestamp, provenance). Exemple :

```json
{
	"0": {"path": "dataset_oiseaux/train/heron/img001.jpg", "class": "heron"},
	"1": {"path": "enregistrements/heron/2026-05-18_14-14-46.jpg", "class": "heron"}
}
```

## 4) Utilisation en inference

- Charger `index.faiss` et `mapping.json` au démarrage si disponibles.
- Pour chaque image, calculer l'embedding via CLIP, normaliser, puis demander les K voisins via FAISS (`index.search`).
- Combiner le résultat FAISS (similarité) et la sortie du classifieur YOLO pour améliorer la décision.

Exemples pratiques :

- Si YOLO renvoie une classe `incertain` mais FAISS trouve un voisin très proche (sim > 0.8), considérer cet élément pour revue humaine prioritaire.
- Si FAISS trouve des voisins multiples avec la même classe, renforcer la confiance locale sur cette classe.

## 5) Mise à jour de l'index

- L'index doit être reconstruit périodiquement si de nouvelles images significatives arrivent (ex: nouvel enregistrement).
- Alternativement, utiliser un index FAISS qui supporte l'ajout dynamique si la latence de construction complète est trop élevée.

## 6) Parametres conseils

- `K=5` voisins par defaut.
- `normalize=True` (L2) avant insertion.
- `index_type=IndexFlatIP` si l'on utilise inner product.
- `nprobe` plus eleve pour des recherches plus precisess mais plus lentes (s'applique aux indexes quantifies).

## 7) Exemple d'API d'appel (pseudocode)

```python
from vector_index import load_index, query

index, mapping = load_index('vectors/index.faiss', 'vectors/mapping.json')
vec = clip_encode(image)
vec = vec / np.linalg.norm(vec)
ids, scores = index.search(vec.reshape(1, -1), k=5)
results = [mapping[str(i)] for i in ids[0]]
```

## 8) Clauses et limitations

- FAISS stocke des vecteurs en memoire; pour des corpus enormes, il faut envisager des indexes compresses ou une architecture distribuée.
- L'usage de CLIP a des biais selon le training data. Valide les resultats sur ton jeu d'images local.

## 9) Intégration dans `analyse_oiseaux.py`

1. Charger l'index s'il existe lors du demarrage du script.
2. Ajouter une option `--vector-interval` pour controler la frequence de calcul des embeddings en mode camera.
3. Ajouter un overlay dans la fenetre video pour afficher les meilleurs voisins et scores.

---

Si tu veux, je peux écrire `vector_index.py` (ou l'améliorer) pour ton jeu de donnees et fournir un exemple complet de pipeline de construction et de requete.

