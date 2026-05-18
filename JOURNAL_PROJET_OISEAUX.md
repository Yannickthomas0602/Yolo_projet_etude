# Journal du projet de reconnaissance d'oiseaux

Ce document garde une trace simple et lisible de la démarche suivie sur le projet. Il résume la préparation du dataset, l'entraînement, la logique métier BDD / INCERTITUDE / HORS_BDD, puis la mise à jour de la documentation.

## 1. Objectif du projet

Le but est de disposer d'un prototype YOLOv5 capable de reconnaître 4 classes d'oiseaux et de servir de base à un système complet de reconnaissance avec décision métier et lecture audio.

## 2. Références utilisées

Les documents de référence du dépôt sont :

- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md)
- [README.md](README.md)
- [classify/tutorial.ipynb](classify/tutorial.ipynb)

## 3. Méthode de classement des images

### 3.1 Dossier source

Les images brutes étaient regroupées dans [Dataset](Dataset) avec 4 sous-dossiers d'origine :

- `balbuzard`
- `heron gris`
- `pixabay_cormorands`
- `pixabay_mouette_goélands`

### 3.2 Classes finales retenues

Le projet a été normalisé autour de 4 classes cibles :

- `balbuzard`
- `heron`
- `cormoran`
- `mouette_goeland`

La classe `mouette_goeland` regroupe volontairement les mouettes et les goélands, car ils étaient confondus dans le périmètre retenu.

### 3.3 Correspondance source vers classe

Le classement a suivi cette logique :

- `balbuzard` -> `balbuzard`
- `heron gris` -> `heron`
- `pixabay_cormorands` -> `cormoran`
- `pixabay_mouette_goélands` -> `mouette_goeland`

### 3.4 Répartition des données

Les images ont été réparties en :

- `train` : 70 %
- `validation` : 20 %
- `test` : 10 %

Le dataset classé a été généré dans [dataset_oiseaux](dataset_oiseaux).

### 3.5 Volume obtenu

Répartition approximative initiale :

- `balbuzard` : 400 images
- `heron gris` : 576 images
- `pixabay_cormorands` : 515 images
- `pixabay_mouette_goélands` : 559 images

Après répartition :

- `train` : 1434 images
- `validation` : 409 images
- `test` : 207 images

## 4. Actions réalisées sur le dépôt

### 4.1 Création du dataset classé

Le dossier [dataset_oiseaux](dataset_oiseaux) a été construit à partir du dossier brut [Dataset](Dataset).

### 4.2 Suppression du dossier brut

Le dossier brut [Dataset](Dataset) a ensuite été supprimé pour éviter de conserver les sources d'origine dans le dépôt.

### 4.3 Mise à jour de la documentation

Les guides ont été harmonisés pour refléter l'état réel du projet :

- 4 classes cibles,
- fusion de mouette et goeland,
- ajout de la logique BDD / INCERTITUDE / HORS_BDD,
- ajout d'un guide de lancement dédié.

### 4.4 Versionnement Git

Le dataset classé, les scripts utilitaires, les poids entraînés et la documentation ont été commités puis poussés sur GitHub.

## 5. Entraînement du modèle

### 5.1 Configuration retenue

L'entraînement a été lancé avec :

- modèle : `yolov5s-cls.pt`
- taille d'image : `224`
- batch size : `32`
- epochs : `30`
- device : `0` sur la RTX 4070

### 5.2 Point de blocage initial

La première tentative d'entraînement a échoué à cause d'un chemin de travail incorrect. Le script cherchait à être lancé depuis `C:\Users\yanni\Desktop\Yolo` au lieu du dossier [yolov5](.).

### 5.3 Correction GPU

Le venv a ensuite été mis à jour avec une version CUDA de PyTorch :

- `torch 2.6.0+cu124`
- `torchvision 0.21.0+cu124`
- `torchaudio 2.6.0+cu124`

La vérification a confirmé :

- `torch.cuda.is_available() = True`
- `torch.cuda.device_count() = 1`
- GPU détecté : `NVIDIA GeForce RTX 4070 Laptop GPU`

### 5.4 Entraînement réussi

L'entraînement de référence a été exécuté avec succès sur le dataset préparé.

- Résultats sauvegardés dans `runs/train-cls/exp4`
- Poids finaux : `runs/train-cls/exp4/weights/best.pt` et `runs/train-cls/exp4/weights/last.pt`

## 6. Logique métier ajoutée

Une logique de décision a été ajoutée dans [classify/predict.py](classify/predict.py) pour distinguer trois états :

- `BDD` : l'espèce est reconnue avec une confiance suffisante,
- `INCERTITUDE` : la confiance est intermédiaire,
- `HORS_BDD` : la confiance est trop faible pour valider la classe.

Cette logique est pilotée par deux seuils :

- seuil BDD : `0.60`
- seuil incertitude : `0.30`

## 7. Lecture audio des cris d'oiseaux (mai 2026)

Une nouvelle fonctionnalité a été ajoutée pour jouer automatiquement le son d'un oiseau après sa détection lors d'une analyse d'image unique.

### 7.1 Objectif

Après la détection et la classification d'un oiseau sur une image, le script [analyse_oiseaux.py](analyse_oiseaux.py) lance automatiquement la lecture d'un fichier audio correspondant à l'espèce détectée.

### 7.2 Fichiers modifiés

- [analyse_oiseaux.py](analyse_oiseaux.py) : ajout des fonctions audio et intégration dans `analyze_single_image()`

### 7.3 Nouvelles fonctions ajoutées

#### `find_audio_file(bird_class: str) -> Optional[Path]`

Recherche un fichier audio correspondant à l'oiseau détecté :
- Mappe la classe normalisée du modèle (`balbuzard`, `heron`, `cormoran`, `mouette_goeland`) vers le dossier audio correspondant dans `cri_predateur_ou_detresse/`
- Cherche tous les fichiers MP3 du dossier
- Sélectionne aléatoirement un fichier MP3 si plusieurs existent
- Retourne `None` si aucun fichier ne correspond (cas des classes inconnues ou mal mappées)

Mappage classe → dossier audio :
- `balbuzard` → `Balbuzard/`
- `heron` → `Heron/`
- `cormoran` → `Cormoran/`
- `mouette_goeland` → `Goeland-mouette/`

#### `play_audio(audio_path: Optional[Path]) -> bool`

Lance la lecture d'un fichier audio MP3 en ouvrant le lecteur système par défaut :
- Sous **Windows** : utilise `os.startfile()` pour ouvrir le fichier avec l'application par défaut
- Sous **Linux** : utilise `xdg-open` via `subprocess`
- Sous **macOS** : utilise `open` via `subprocess`
- Retourne `True` si la lecture a réussi, `False` sinon
- Gère les erreurs d'exécution et les cas où le fichier n'existe pas

#### `play_bird_audio(bird_class: str) -> bool`

Fonction façade qui orchestre l'ensemble du processus de lecture audio :
1. Recherche un fichier audio pour l'oiseau
2. Sélectionne un son aléatoirement
3. Lance la lecture automatique

## 8. Classement automatique des images analysées (mai 2026)

Le script [analyse_oiseaux.py](analyse_oiseaux.py) copie maintenant les images analysées dans un répertoire dédié selon le résultat obtenu.

Règle actuelle :

- classe reconnue en BDD -> dossier spécifique à l'espèce, par exemple [enregistrements/heron](enregistrements),
- INCERTITUDE -> [enregistrements/incertitude](enregistrements),
- HORS_BDD -> [enregistrements/autre](enregistrements).

L'objectif est de garder une trace visuelle simple des tests, de faciliter les vérifications manuelles et de conserver séparés les cas sûrs, douteux et hors dataset.
4. Retourne le statut de la lecture

### 7.4 Intégration dans `analyze_single_image()`

La fonction a été modifiée pour :
1. Lancer la prédiction YOLOv5 comme avant
2. Afficher les résultats en console
3. **[NOUVEAU]** Lancer automatiquement la lecture audio si l'oiseau est reconnu (`BDD` ou `INCERTITUDE`)
4. Afficher un message d'information si l'oiseau n'est pas reconnu (`HORS_BDD`)
5. Générer et afficher le graphique de confiance

Ordre d'exécution pour une image unique :
```
Image analysée → YOLOv5 prédiction → Affichage console → Lecture audio (si reconnue) → Graphique
```

### 7.5 Gestion des cas d'erreur

Le système gère gracieusement les situations suivantes :

- **Oiseau non mappé** : message de log `[AUDIO]` en jaune expliquant qu'aucun dossier audio n'existe pour cette classe
- **Dossier audio manquant** : avertissement `[AUDIO]` indiquant le chemin introuvable
- **Aucun fichier MP3 trouvé** : log `[AUDIO]` expliquant qu'aucun MP3 n'existe pour cet oiseau
- **Fichier audio corrompu ou inaccessible** : l'erreur système est affichée en détail, la lecture échoue silencieusement
- **Classe `HORS_BDD`** : pas de lecture audio (l'oiseau n'est pas reconnu), message informatif affiché
- **Plateforme non supportée** : message `[AUDIO]` jaune indiquant que la plateforme du système n'est pas supportée

### 7.6 Logs et traçabilité

Chaque opération audio est loggée explicitement en console avec codes couleur :

- **Cyan** : messages informatifs (`[AUDIO] Recherche d'un son...`, `[AUDIO] Sélectionné...`)
- **Vert** : succès (`[AUDIO] Lecture lancée...`)
- **Jaune** : avertissements et cas spéciaux (`[AUDIO] Classe non mappée...`, `[AUDIO] Aucun fichier MP3...`)
- **Rouge** : erreurs (`[AUDIO] Fichier introuvable...`, `[AUDIO] Erreur lors de la lecture...`)

Exemple de log complet :
```
[AUDIO] Recherche d'un son pour 'balbuzard'...
[AUDIO] Sélectionné pour 'balbuzard': balbuzard_preda_1.mp3
[AUDIO] Lecture lancée: balbuzard_preda_1.mp3
```

### 7.7 Comportement en mode dossier (batch)

La fonction `analyze_folder()` **ne déclenche pas de lecture audio** pour éviter :
- Les interruptions répétées lors du traitement batch
- Les consommations de ressources système non nécessaires
- Les bruits gênants durant une analyse longue sur de nombreuses images

La lecture audio est réservée à l'analyse interactive d'une **image unique**.

### 7.8 Dépendances et compatibilité

- Aucune dépendance supplémentaire requise
- Utilise uniquement `os`, `subprocess`, `sys` et `random` (modules standards Python)
- Compatible **Windows 10+**, **Linux** (avec gestionnaire d'applications configuré), **macOS 10.12+**
- Les fichiers audio dans `cri_predateur_ou_detresse/` doivent être en format **MP3** (extension `.mp3`)

### 7.9 Limites et notes futures

1. **Lecture asynchrone** : la fonction est non-bloquante sur Windows (le lecteur s'ouvre en arrière-plan) mais peut bloquer sur d'autres OS selon le lecteur par défaut configuré
2. **Bruit de fond** : dans un environnement batch, il est possible que plusieurs lecteurs s'ouvrent simultanément ; ce cas ne se produit que si `analyze_folder()` était modifiée pour lancer la lecture audio
3. **Qualité audio** : aucune normalisation ou ajustement de volume n'est appliqué ; le son est joué au volume par défaut du système
4. **Sélection aléatoire** : la sélection des fichiers MP3 ne garantit pas une distribution uniforme sur plusieurs analyses ; pour un test statistique rigoureux, une semence (`seed`) ou un log des fichiers joués peut être ajouté
5. **Extension `.MP3`** : les fichiers en majuscules `.MP3` ne sont actuellement pas reconnus (case-sensitive sur Linux) ; amélioration future possible

### 7.10 Exemple d'utilisation

```bash
# Lancer l'analyse interactive
python analyse_oiseaux.py

# Choisir l'option 1 (analyser une seule image)
# Votre choix (1/2) : 1

# Entrer le chemin vers l'image
# Chemin complet de l'image : C:\path\to\heron.jpg

# Résultat attendu :
# - Affichage de la prédiction en console
# - Ouverture du lecteur MP3 système avec le cri d'un héron
# - Génération du graphique de confiance
# - Affichage du chemin vers le graphique enregistré
```

### 7.11 Tests réalisés


### 7.12 Signal d'alerte pour l'INCERTITUDE (canonmp3)

Une amélioration a été apportée pour différencier la lecture audio selon le statut de confiance :

#### Logique de lecture audio raffinée

La fonction `analyze_single_image()` a été modifiée pour jouer des sons différents selon le statut :

- **BDD (confiance >= 60%)** : joue le cri authentique de l'oiseau détecté
	- Exemple : détection d'un héron → lecture d'un cri de héron
	- Fichier sélectionné aléatoirement parmi les cris disponibles pour cette espèce

- **INCERTITUDE (confiance 50-60%)** : joue un signal d'alerte générique (`canon.mp3`)
	- Signal sonore pour indiquer que le modèle hésite
	- Permet d'alerter l'utilisateur que le résultat est incertain
	- Localisation : `cri_predateur_ou_detresse/canon.mp3`

- **HORS_BDD (confiance < 50%)** : pas de son, uniquement un message informatif
	- Oiseau non reconnu par le modèle
	- Aucune lecture audio pour ne pas générer de fausse alerte

#### Nouvelle constante

```python
# Fichier audio d'alerte (doute) - joué en cas d'INCERTITUDE
AUDIO_ALERT = AUDIO_DIR / "canon.mp3"
```

#### Exemple de flux complet

```
Analyse d'une image de héron avec confiance 55% (INCERTITUDE)
	↓
YOLOv5 prédiction : "heron", score 55%
	↓
Affichage console : classe = "heron", confiance = 55%, statut = "INCERTITUDE"
	↓
[AUDIO] Statut INCERTITUDE détecté - lecture du signal d'alerte...
	↓
Ouverture du lecteur système avec canon.mp3
	↓
Génération du graphique de confiance
```

#### Avantages de cette approche

1. **Feedback clair** : L'utilisateur sait immédiatement si le résultat est certain ou douteux
2. **Pas de confusion** : Un cri d'oiseau faux n'est jamais joué (cas d'INCERTITUDE = son d'alerte, cas d'HORS_BDD = silence)
3. **Modulable** : Le signal d'alerte peut être changé facilement (remplacer `canon.mp3` par un autre fichier)
4. **Non-bloquant** : La lecture audio n'interrompt pas le processus d'analyse

#### Tests validés pour la version avec signal d'alerte

- ✓ Statut BDD : son de l'oiseau détecté
- ✓ Statut INCERTITUDE : canon.mp3 joué avec message explicite
- ✓ Statut HORS_BDD : pas de son, message d'information
- ✓ Logging détaillé : chaque action de lecture est loggée en jaune/cyan/vert/rouge
- ✓ Aucun impact sur le mode batch

En pourcentage, cela correspond à 60 % pour BDD et 30 % pour l'incertitude.

## 7. Validation de l'inférence

Un test a été réalisé sur [pixabay_91570.jpg](dataset_oiseaux/train/cormoran/pixabay_91570.jpg).

Résultat observé :

- prédiction principale : `cormoran`
- confiance élevée
- statut métier validé selon le seuil choisi

Le comportement a aussi été vérifié avec des seuils plus stricts pour forcer les sorties `INCERTITUDE` puis `HORS_BDD`.

## 8. Mise à jour de la documentation

Les documents Markdown ont été remis en cohérence avec l'état actuel du projet.

### 8.1 Fichiers concernés

- [GUIDE_OISEAUX.md](GUIDE_OISEAUX.md)
- [SETUP_ENTRAINEMENT_OISEAUX.md](SETUP_ENTRAINEMENT_OISEAUX.md)
- [LANCER_IA_OISEAUX.md](LANCER_IA_OISEAUX.md)

### 8.2 Ce qui a été clarifié

- la démarche complète du projet,
- les chemins réels pour lancer l'IA,
- les commandes pour l'entraînement et l'inférence,
- la règle métier BDD / incertitude / hors BDD,
- le guide court de lancement.

## 9. État actuel du projet

À ce stade :

- le dataset brut n'existe plus dans le dépôt,
- le dataset classé est disponible dans [dataset_oiseaux](dataset_oiseaux),
- un modèle de référence est disponible dans `runs/train-cls/exp4`,
- la logique de décision est déjà intégrée,
- la documentation est regroupée autour d'un guide principal, d'un guide d'installation et d'un guide de lancement.

## 10. Prochaine étape possible

La prochaine étape naturelle est de brancher la décision métier sur la lecture audio et d'industrialiser le test final sur 1000 images variées.

## 11. Analyse interactive ajoutée

Un script dédié a été ajouté pour simplifier les analyses d'images sans retaper la commande YOLOv5 complète.

## Mises à jour récentes (18 mai 2026)

Résumé des modifications récentes intégrées au dépôt :

- Mode caméra : ajout d'un mode caméra réactif utilisant un thread d'acquisition, avec cooldown pour éviter l'analyse en continu et sauvegarde des images analysées.
- Gestion propre de l'interruption (Ctrl-C) : recommandations et nettoyages (`cap.release()`, `cv2.destroyAllWindows()`, arrêt des threads) pour éviter les warnings au shutdown.
- Lecture audio : intégration complète pour la lecture automatique des cris (fichiers MP3) lors d'une analyse d'image unique, avec `canon.mp3` en cas d'INCERTITUDE.
- Classement des images : les images analysées sont copiées dans `enregistrements/` selon le statut (`BDD/`, `INCERTITUDE/`, `HORS_BDD/`).
- Intégration Azure (optionnelle) : ajout de `transferer_donnees_azure()` puis bascule vers `scripts/azure_transfer.py` pour envoyer l'image vers `archives-photos` et le JSON vers `archives-json`. Le JSON inclut `heure` au format `HH:MM`, `statut` et une `action` dérivée du statut métier. Les imports Azure sont dynamiques pour tolérer l'absence de SDK.
- Installation SDKs : `azure-storage-blob` et `azure-iot-device` installés dans l'environnement virtuel pour les tests locaux.
- Prototype vectoriel : ajout d'un prototype CLIP + FAISS (`vectors/index.faiss` attendu) + script de construction prévu dans `scripts/` (index hors-ligne).
- Documentation enrichie : ajouts de schémas Mermaid dans `EXPLICATION_PRINCIPES.md` (architecture, flux Azure, flux vectoriel) et nettoyage pour compatibilité des parsers Mermaid.
- Sécurité : note importante sur la rotation des clés Azure si des secrets ont été exposés accidentellement et recommandation d'utiliser `.env` non suivi par git.

Si tu veux, je peux :

- automatiser le build de l'index vectoriel (`scripts/build_vector_index.py`) ;
- générer les images SVG/PNG des diagrammes Mermaid dans `docs/images/` ;
- committer ces modifications et ouvrir une PR si tu veux un historique distinct.


### 11.1 Nouveau point d'entrée

- [analyse_oiseaux.py](analyse_oiseaux.py)

### 11.2 Ce qu'il fait

- demande si l'utilisateur veut analyser une image ou un dossier complet,
- lance automatiquement [classify/predict.py](classify/predict.py) avec les poids de référence,
- récupère et affiche les probabilités de prédiction,
- génère un graphique propre pour une image unique,
- ouvre automatiquement le graphique après génération,
- accepte aussi les dossiers organisés en sous-dossiers récursifs,
- calcule la réussite globale d'un dossier,
- affiche les classes les plus souvent confondues,
- sauvegarde les graphiques et un résumé JSON dans `results/`,
- installe automatiquement `matplotlib`, `tqdm` et `colorama` si nécessaire.

Note : pour les analyses de dossier, le script évite d'enregistrer les images annotées individuellement afin d'accélérer le traitement (option `--nosave`) ; le produit principal est le fichier de synthèse JSON et les graphiques enregistrés dans `results/`.

### 11.3 Résultat attendu

Le script fournit désormais une interface console interactive plus rapide pour tester le modèle sur une image ou sur un dataset entier, y compris quand le dataset contient des sous-dossiers par classe, tout en gardant les sorties exploitables pour le suivi projet.

## 12. Mise à jour des règles de décision (mai 2026)

Le projet a été aligné sur une règle de décision plus simple dans [analyse_oiseaux.py](analyse_oiseaux.py) :

- la décision `BDD` / `INCERTITUDE` / `HORS_BDD` se fait uniquement sur le score top-1,
- `BDD` si top-1 >= 0.60,
- `INCERTITUDE` si 0.50 <= top-1 < 0.60,
- `HORS_BDD` si top-1 < 0.50.

Cette mise à jour garantit qu'un exemple comme `cormoran = 43 %` est bien classé `HORS_BDD`.

Le modèle de référence utilisé pour les analyses est désormais :

- `runs/train-cls/exp_retrain/weights/best.pt`.
