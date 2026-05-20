# Analyse vectorielle — explication simple et sans maths

Cette page explique, avec des mots simples et des images mentales, ce qu'est l'analyse vectorielle et comment elle aide à reconnaître des photos d'oiseaux dans ce projet.

## Idée de base — l'empreinte digitale des images

Imagine que chaque photo a une petite "fiche" invisible qui la décrit (couleurs dominantes, formes, textures). Cette fiche n'est pas du texte lisible par un humain, ce sont juste des nombres que l'ordinateur utilise — on appelle ça un "embedding".

- Photo → on calcule sa fiche (embedding)
- Deux photos similaires ont des fiches très proches (comme deux empreintes digitales qui se ressemblent)

Donc : au lieu de comparer les images pixel par pixel, on compare leurs fiches. C'est plus rapide et ça marche même si la photo est un peu différente (angle, lumière...).

## Qui fait quoi ? (sans jargon)

- CLIP (ou un modèle similaire) : c'est l'outil qui regarde une image et fabrique sa fiche.
- FAISS : c'est la grosse bibliothèque qui range toutes les fiches et trouve très vite celles qui se ressemblent.

Image simple : CLIP = le photographe qui transforme la photo en description, FAISS = le bibliothécaire qui retrouve les livres voisins dans un catalogue.

## Exemple concret pas à pas

1. Tu prends une photo d'un oiseau.
2. Le système calcule la fiche de cette photo (quelques milliers de chiffres invisibles).
3. FAISS cherche dans sa base les fiches déjà connues qui ressemblent le plus.
4. Le système renvoie les photos proches et un score de similarité (plus le score est haut, plus les photos se ressemblent).
5. On combine ce résultat avec ce que le classifieur YOLOv5 dit pour décider si c'est bien la même espèce.

Exemple d'interprétation :
- Si la similarité est très forte et que YOLO dit aussi "héron", on accepte "héron".
- Si YOLO est incertain mais FAISS trouve plusieurs photos très proches étiquetées "héron", on met "héron" en suggestion forte.
- Si tout est faible, on signale "hors base" et on demande une revue humaine.

## Pourquoi c'est utile ?

- Rapide : FAISS retrouve des voisins en un clin d'œil, même dans des milliers d'images.
- Robuste : aide quand le classifieur est incertain (mêmes formes, mêmes textures).
- Pratique : on peut retrouver des images déjà vues sur le terrain pour vérifier des anomalies.

## Limites faciles à comprendre

- Si la base contient de mauvaises étiquettes, FAISS va retrouver des mauvaises correspondances.
- Les photos très différentes (lumière, angle) peuvent donner des fiches différentes → moins de similarité.
- FAISS stocke des données en mémoire : pour des millions de photos il faut prévoir plus de ressources.

## Où voir / tester dans ce projet

- Construire l'index (hors-ligne) : `python vector_index.py --build --sources dataset_oiseaux enregistrements --out vectors`
- Tester une requête : le script `analyse_oiseaux.py` affiche les voisins visuels si `vectors/index.faiss` existe.

## En résumé

L'analyse vectorielle, c'est convertir les images en petites fiches numériques, puis chercher les fiches les plus proches dans une grande armoire (FAISS). C'est un moyen puissant d'aider l'IA principale quand elle doute.

---

Veux-tu que je replace ce texte dans `README.md` ou que je l'intègre dans `analyse_oiseaux.py` comme aide à l'utilisateur ?
