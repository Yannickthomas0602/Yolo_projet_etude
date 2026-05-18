# Mode vectoriel (CLIP + FAISS)

Cette page explique comment construire un index vectoriel d'images et l'utiliser depuis `analyse_oiseaux.py` pour améliorer la décision métier (détecter si une image est proche d'images connues du dataset).

## Principe

- On extrait un embedding (vecteur) pour chaque image du dataset (ou des images déjà enregistrées) à l'aide d'un modèle d'embeddings visuels (ici CLIP). 
- On indexe ces vecteurs avec FAISS pour effectuer des recherches rapides de voisins les plus proches.
- Lors d'une analyse d'image (mode image ou caméra), on calcule l'embedding de l'image et on interroge l'index.
- Si le voisin le plus proche est suffisamment similaire (distance élevée / inner product proche de 1), on peut en déduire que l'image est « proche » d'une image BDD et adapter la décision métier.

## Fichiers ajoutés

- `vector_index.py` : outil pour construire l'index (`--build`) et pour interroger (`query_image`).
- `vectors/` : répertoire cible attendu pour l'index (`index.faiss`) et la `mapping.json`.

## Installation des dépendances

Le script tente d'installer automatiquement les dépendances s'il les trouve manquantes :

- CLIP (OpenAI), FAISS CPU, Pillow, tqdm.

Si tu préfères installer manuellement :

```powershell
.\.venv\Scripts\Activate.ps1
pip install git+https://github.com/openai/CLIP.git faiss-cpu Pillow tqdm
```

## Construire l'index

Exemple :

```powershell
python vector_index.py --build --sources dataset_oiseaux enregistrements --out vectors
```

Le script parcourra les dossiers `dataset_oiseaux` et `enregistrements`, extraira les embeddings, construira l'index et sauvegardera :

- `vectors/index.faiss`
- `vectors/mapping.json`

## Utilisation dans `analyse_oiseaux.py`

- Si `vectors/index.faiss` est présent, `analyse_oiseaux.py` interroge automatiquement l'index après l'analyse d'une image et affiche les voisins visuels (chemin + score). 
- Le score est un inner product sur vecteurs normalisés (équivalent approximatif du cosinus). Plus il est proche de 1.0, plus la similitude est élevée.

## Stratégie d'intégration recommandée

- Phase offline : construire l'index régulièrement (après avoir collecté de nouvelles images dans `enregistrements`).
- Phase online : lors d'une prédiction faible (HORS_BDD ou INCERTITUDE), interroger l'index ; si similitude > seuil (ex: 0.3–0.5 selon calibration), considérer la piste BDD et enregistrer l'information pour revue humaine.

## Calibration

Calibre le seuil en analysant les distances entre images connues (intra-classe) et images hors-BDD sur un jeu de validation. Commence par 0.30 et ajuste vers 0.50 si trop de faux positifs.
