# Mode camera pour analyse_oiseaux.py

Ce document décrit comment ajouter un troisieme mode d'analyse dans [analyse_oiseaux.py](analyse_oiseaux.py) :

1. analyse d'une image unique,
2. analyse d'un dossier d'images,
3. analyse en direct depuis la camera de l'ordinateur.

Le but est de garder la logique actuelle du projet, deja documentee dans [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md), [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md) et [JOURNAL_PROJET_OISEAUX.md](JOURNAL_PROJET_OISEAUX.md), tout en ajoutant un mode temps reel simple a utiliser et facile a maintenir.

## 1. Objectif fonctionnel

Le script doit proposer un choix clair a l'utilisateur au lancement :

- analyser une image locale,
- analyser tout un dossier d'images,
- ouvrir la camera de l'ordinateur et lancer les predictions image par image.

Le mode camera doit reutiliser la meme logique metier que les autres modes :

- meme modele YOLOv5,
- meme seuil BDD / INCERTITUDE / HORS_BDD,
- meme affichage console,
- meme lecture audio pour une image unique si le projet la conserve.

## 2. Principe general recommande

La solution la plus simple consiste a separer le script en 3 couches :

- une couche d'acquisition d'image,
- une couche d'inference,
- une couche d'affichage et de decision metier.

En pratique, le mode camera ne doit pas dupliquer la logique de classification. Il doit seulement fournir une image courante au moteur d'analyse existant.

Schema conseille :

~~~text
camera -> frame -> image temporaire -> run_yolov5_prediction -> post-traitement -> affichage
~~~

Cette approche permet de garder le code coherent avec le reste du fichier et de limiter les risques de regression.

## 3. Etat actuel du script

Le fichier [analyse_oiseaux.py](analyse_oiseaux.py) gere deja :

- la recherche d'images dans un dossier,
- le lancement du classifieur YOLOv5 en sous-processus,
- le post-traitement des resultats,
- l'affichage console,
- la synthese dossier,
- la lecture audio pour une image unique.

La fonction centrale a reutiliser est [run_yolov5_prediction](analyse_oiseaux.py), qui prend actuellement un chemin vers une image ou une liste d'images. Le mode camera peut s'appuyer dessus en ecrivant chaque frame dans un fichier temporaire avant de lancer l'analyse.

## 4. Architecture proposee pour la camera

### 4.1 Nouvelles fonctions a ajouter

Tu peux ajouter trois fonctions principales :

- open_camera(index: int = 0) -> ... pour ouvrir la camera,
- analyze_camera_stream(...) -> None pour lire les frames et lancer l'inference,
- save_frame_to_tempfile(frame) -> Path pour enregistrer temporairement une image exploitable par YOLOv5.

Si tu veux garder le code plus compact, une seule fonction de haut niveau peut aussi suffire :

~~~text
analyze_camera()
~~~

Mais il reste preferable de separer l'ouverture de la camera, la boucle de lecture et la conversion en fichier temporaire.

### 4.2 Dependances usuelles

Le mode camera necessite en pratique OpenCV.

Le projet utilise deja des dependances dynamiques dans [analyse_oiseaux.py](analyse_oiseaux.py), donc tu peux :

- ajouter opencv-python aux dependances si ce n'est pas deja present,
- verifier sa disponibilite avec la meme logique que pour les autres modules,
- afficher un message clair si la camera ou OpenCV ne sont pas accessibles.

### 4.3 Format de la camera

La camera doit etre ouverte avec l'index 0 par defaut, ce qui correspond generalement a la camera principale.

Il peut etre utile de prevoir un parametre optionnel si l'ordinateur possede plusieurs cameras :

~~~text
python analyse_oiseaux.py --camera 1
~~~

## 5. Flux d'execution recommande

Le mode camera peut suivre le cycle suivant :

1. l'utilisateur choisit le mode camera,
2. le script ouvre la camera,
3. chaque frame est capturee,
4. une image est enregistree temporairement,
5. YOLOv5 analyse cette image,
6. le script affiche la prediction et le statut,
7. un appui sur une touche stoppe la boucle.

Pour eviter de saturer le systeme, il est recommande de ne pas lancer une inference sur chaque frame brute sans pause. Il vaut mieux :

- analyser une frame sur plusieurs,
- ou ajouter un delai court entre deux analyses,
- ou ne traiter qu'une frame apres appui clavier.

## 6. Interface utilisateur

### 6.1 Menu interactif

Si analyse_oiseaux.py repose sur un menu terminal, la meilleure evolution est d'ajouter un troisieme choix :

~~~text
1. Analyser une image
2. Analyser un dossier
3. Analyser depuis la camera
0. Quitter
~~~

Ce menu doit rester simple, car le but est d'avoir un lancement rapide pour des tests de terrain.

### 6.2 Arguments optionnels

Si tu veux aussi un usage non interactif, tu peux ajouter des arguments argparse :

- --mode image
- --mode folder
- --mode camera
- --camera-index 0

Cette option est utile si tu veux automatiser les essais ou lancer le script depuis un raccourci.

## 7. Choix technique pour l'analyse camera

### Option A - Analyse ponctuelle a la demande

Le principe : une image est capturee quand l'utilisateur appuie sur une touche, puis elle est envoyee a YOLOv5.

Avantages :

- peu de calcul,
- comportement stable,
- plus simple a debugguer.

Inconvenients :

- moins fluide qu'un vrai flux temps reel,
- demande une action utilisateur pour chaque capture.

### Option B - Analyse continue avec affichage

Le principe : la camera tourne en boucle et chaque frame ou sous-ensemble de frames est analyse.

Avantages :

- experience plus naturelle,
- plus proche d'une application en direct.

Inconvenients :

- plus lourd,
- peut ralentir rapidement si l'inference est trop frequente,
- risque de multiplier les fichiers temporaires si la gestion n'est pas propre.

Pour ce projet, l'option A est la plus robuste pour une premiere version. L'option B peut venir ensuite si besoin.

## 8. Points d'attention dans analyse_oiseaux.py

### 8.1 Gestion des fichiers temporaires

Comme [run_yolov5_prediction](analyse_oiseaux.py) travaille avec des chemins de fichiers, la camera devra generer une image temporaire au format jpg ou png.

Il faut veiller a :

- supprimer le fichier temporaire apres usage,
- fermer correctement la camera,
- gerer les erreurs si la lecture de frame echoue.

### 8.2 Gestion des resultats audio

La lecture audio est adaptee a une image unique. En mode camera, il faut eviter de relancer le son en boucle sur la meme espece si la camera analyse plusieurs frames successives.

Bonne pratique :

- ne jouer le son que si la classe detectee change,
- ou ne jouer le son qu'au premier resultat pertinent,
- ou ajouter un cooldown temporel entre deux lectures audio.

### 8.3 Performance

Le mode camera sera plus fluide si tu utilises :

- un modele leger,
- une resolution moderee,
- une frequence d'inference limitee,
- eventuellement un resize avant sauvegarde temporaire.

## 9. Proposition de comportement utilisateur

Le comportement attendu peut etre le suivant :

- en mode image : analyse d'un seul fichier,
- en mode dossier : analyse recursive et bilan global,
- en mode camera : ouverture de la camera, prediction sur demande ou en continu, arret avec une touche clavier.

Exemple d'usage cible :

~~~text
python analyse_oiseaux.py
~~~

Puis choix dans le menu :

~~~text
3
~~~

## 10. Classement des images analysées

Chaque image capturée ou analysée est copiée automatiquement dans un sous-dossier du répertoire [enregistrements](enregistrements).

La règle est volontairement simple pour rendre le fonctionnement facile à relire et à contrôler après coup :

- si le modèle reconnaît un héron en BDD, l'image est copiée dans [enregistrements/heron](enregistrements),
- si le score est en zone d'incertitude, l'image est copiée dans [enregistrements/incertitude](enregistrements),
- si l'image est classée hors BDD, elle est copiée dans [enregistrements/autre](enregistrements).

Le nom du dossier BDD dépend de la classe détectée. Par exemple :

- `balbuzard` -> `enregistrements/balbuzard`
- `heron` -> `enregistrements/heron`
- `cormoran` -> `enregistrements/cormoran`
- `mouette_goeland` -> `enregistrements/mouette_goeland`

Ce choix permet de séparer proprement les cas d'usage :

- les images reconnues servent de preuve de détection,
- les images en incertitude servent à revoir les cas limites,
- les images hors BDD servent à enrichir le retour terrain ou à repérer des erreurs de modèle.

## 11. Etapes de mise en place recommandees

1. Ajouter la dependance OpenCV si elle manque.
2. Ajouter le choix camera dans le menu ou dans argparse.
3. Creer une fonction de capture camera.
4. Ecrire chaque frame dans un fichier temporaire.
5. Reutiliser [run_yolov5_prediction](analyse_oiseaux.py) sans dupliquer la logique de classification.
6. Afficher le resultat avec les memes seuils metiers que les autres modes.
7. Tester sur une camera interne puis sur une camera externe si besoin.

## 12. Tests a valider

Avant de considerer le mode camera comme termine, il faut verifier :

- la camera s'ouvre correctement,
- la camera se ferme proprement,
- une frame capturee est bien analysable par YOLOv5,
- les resultats console restent lisibles,
- la lecture audio ne boucle pas sans controle,
- les fichiers temporaires sont nettoyes.

## 13. Integration documentaire

Si tu veux rendre cette fonctionnalite plus visible dans la documentation existante, ajoute ensuite un lien vers ce document depuis :

- [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md),
- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md),
- [JOURNAL_PROJET_OISEAUX.md](JOURNAL_PROJET_OISEAUX.md).

Cela permettra de garder une entree courte pour le lancement, et une page detaillee pour la camera.

## 14. Analyse vectorielle continue (CLIP + FAISS)

Principe :

- On utilise CLIP (modèle de représentation visuelle) pour convertir une image en un vecteur d'embedding haute-dimension.
- Ces embeddings sont normalisés puis indexés avec FAISS (index vecteur rapide) pour permettre des recherches des plus proches voisins (ANN).
- En mode caméra continue, on calcule périodiquement l'embedding de la frame courante et on interroge l'index FAISS pour récupérer les images visuellement proches (voisins).

Pourquoi c'est utile :

- Permet de retrouver rapidement des images similaires issues d'un jeu de référence (ex: `dataset_oiseaux` + `enregistrements`).
- Utile pour détecter des occurrences récentes d'une même espèce ou repérer des faux positifs fréquents.
- Peut être combiné avec la classification (règles métier) : si le classifieur est incertain mais le plus proche voisin a une similarité élevée, on peut considérer cela comme un signal d'appui.

Comportement en pratique :

- Le script charge l'index FAISS et le modèle CLIP une seule fois au démarrage de la caméra.
- Pour limiter la charge GPU/CPU, l'embedding est calculé toutes les N secondes (par défaut 2s). L'utilisateur peut ajuster ce délai dans le code.
- Les voisins et leurs scores sont dessinés en overlay sur la fenêtre vidéo pour feedback immédiat.
- Option avancée : déclencher automatiquement une capture/enregistrement quand la similarité dépasse un seuil calibré (ex: 0.30). Ceci n'est pas activé par défaut mais peut être ajouté dans `analyse_oiseaux.py`.

Comment construire l'index :

1. Active l'environnement Python du projet.
2. Lance la commande suivante depuis le dossier `yolov5` :

```bash
python vector_index.py --build --sources dataset_oiseaux enregistrements --out vectors
```

3. Le dossier `vectors` contiendra `index.faiss` et `mapping.json`. Au prochain lancement de la caméra, l'analyse vectorielle continue s'activera automatiquement si l'index est présent.

Remarques de performance :

- CLIP encode les images : sur CPU cela peut être lent ; préférez un GPU si disponible.
- Pour des flux en temps réel, réduisez la résolution d'entrée ou augmentez l'intervalle `vector_interval`.
- Pour un usage embarqué, envisagez des modèles plus légers (distil-CLIP ou Mobile-compatible).

