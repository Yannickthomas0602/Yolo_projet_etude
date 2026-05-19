# Guide complet du projet de reconnaissance d'oiseaux

Ce document rassemble les explications, les bonnes pratiques et les commandes utiles pour construire un système complet de reconnaissance d'oiseaux avec ce dépôt YOLOv5. Le prototype actuel est déjà organisé autour des 4 classes cibles, du dataset classé et d'une logique de décision BDD / incertitude / hors BDD.

Si tu veux lancer directement le modèle ou tester une image, commence par [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md). Si tu veux suivre la démarche complète, lis aussi [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md) et [JOURNAL_PROJET_OISEAUX.md](JOURNAL_PROJET_OISEAUX.md).

Le projet final doit couvrir trois étapes:

1. détecter l'oiseau dans une image ou un flux caméra,
2. identifier son espèce,
3. jouer automatiquement le son correspondant.

Contraintes fonctionnelles du prototype:

- le modèle doit gérer exactement 4 classes cibles,
- les sons sont déjà attribués à chaque espèce,
- les sons d'une même espèce doivent être joués aléatoirement,
- en test final, on évalue le comportement sur 1000 images d'oiseaux variés.

Le dépôt contient deux voies différentes:

- la détection d'objets avec YOLOv5, utilisée pour localiser l'oiseau dans l'image,
- la classification d'images, utile pour reconnaître l'espèce à partir d'une image déjà recadrée.

Le README du dépôt sépare clairement ces deux usages, avec une section détection et une section classification dans [README.md](README.md) et un notebook de classification dans [classify/tutorial.ipynb](classify/tutorial.ipynb).

## 1. Quel pipeline choisir

Pour un projet d'espèces d'oiseaux en temps réel, l'architecture la plus robuste est la suivante:

1. un détecteur localise l'oiseau dans l'image,
2. un classifieur identifie l'espèce sur le recadrage de l'oiseau,
3. une couche applicative joue le son associé à l'espèce détectée.

Cette séparation a plusieurs avantages:

- le détecteur apprend à trouver les oiseaux même dans des scènes complexes,
- le classifieur apprend à distinguer les espèces avec des images centrées sur l'animal,
- l'application finale reste plus simple à faire évoluer.

Si ton dataset contient uniquement une image par espèce, sans annotations de boîtes, tu peux commencer par la classification. Si tu veux détecter un oiseau dans une scène avec arrière-plan, il faudra annoter des boîtes de détection.

## 1.1 Démarche suivie dans ce dépôt

La démarche appliquée dans ce projet est simple et progressive :

1. classer les images du dossier brut dans 4 espèces finales,
2. normaliser les noms de classes pour éviter les problèmes de chemins,
3. séparer le dataset en train, validation et test,
4. entraîner un premier classifieur YOLOv5 sur GPU,
5. ajouter une logique métier pour distinguer `BDD`, `INCERTITUDE` et `HORS_BDD`,
6. valider l'inférence sur une image connue,
7. documenter la procédure pour pouvoir relancer le modèle sans ambiguïté.

L'objectif n'est pas seulement de produire un modèle, mais aussi de garder une trace claire de chaque étape pour qu'un futur lancement soit reproductible.

## 2. Gestion du dataset

Tu as déjà environ 650 images par espèce. C'est une bonne base pour démarrer, à condition de nettoyer le dataset avant l'entraînement.

Dans cette version, limite le périmètre à 4 classes exactement, mais garde en tête qu'à ce stade le dataset n'est pas encore complet. Le but est d'obtenir un prototype fiable et simple à valider, puis de compléter les classes manquantes.

### 2.1 Trier et nettoyer les images

Garde toujours une copie brute intacte du dataset original, puis travaille sur une copie propre.

À supprimer ou isoler:

- les images floues,
- les images trop sombres ou trop surexposées,
- les doublons exacts,
- les quasi-doublons provenant d'une rafale ou d'une vidéo,
- les images avec plusieurs oiseaux si tu fais de la classification pure,
- les images dont l'espèce est incertaine,
- les images où l'oiseau est trop petit, coupé ou masqué.

Bonnes pratiques utiles:

- vérifier manuellement un échantillon de chaque espèce,
- séparer les images provenant de la même source avant de faire le split train/validation/test,
- ne pas mélanger les images très proches d'une même scène entre les différents splits,
- supprimer les images mal étiquetées avant l'entraînement,
- garder des images variées en angle, lumière, fond et distance.

### 2.2 Organisation des dossiers

Si tu fais de la classification, la structure recommandée est celle-ci:

```txt
dataset/
├── train/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
│   └── aigrette/
├── validation/
│   ├── moineau/
│   ├── corbeau/
│   ├── pigeon/
│   ├── heron/
│   ├── merle/
    └── aigrette/
└── test/
    ├── moineau/
    ├── corbeau/
    ├── pigeon/
    ├── heron/
    ├── merle/
    └── aigrette/
```

Cette structure correspond au mode classification de YOLOv5, comme indiqué dans [README.md](README.md#L364) et dans [data/ImageNet.yaml](data/ImageNet.yaml).
