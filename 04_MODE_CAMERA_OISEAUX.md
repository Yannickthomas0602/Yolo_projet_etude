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

... (contenu abrégé pour lisibilité)
