 # Comprendre le système d'analyse d'oiseaux (explication technique simple)

Ce texte s'adresse à quelqu'un qui n'est pas technique mais souhaite comprendre comment fonctionne le système. J'explique chaque étape avec des mots simples et des analogies faciles.

## Vue d'ensemble

Le système fait trois choses principales :

- Il prend des photos (la caméra).
- Il regarde les photos et propose une espèce (l'IA).
- Il range les photos et aide à retrouver des images semblables (stockage et recherche visuelle).

## Détail des étapes — en termes simples

1. La caméra filme en direct. On voit la vidéo dans une fenêtre.
2. Quand on appuie sur `c`, on prend une photo (appelée "frame").
3. La photo est temporairement enregistrée sur l'ordinateur pour traitement.
4. Le programme envoie la photo à l'IA : l'IA est comme un expert qui a appris en regardant beaucoup de photos auparavant.
5. L'IA renvoie sa meilleure proposition (par ex. "héron") et un pourcentage qui indique à quel point elle est sûre.
6. Selon ce pourcentage, le programme classe la photo et l'enregistre dans un dossier adapté :
   - si l'IA est sûre → dossier de l'espèce,
   - si elle doute → dossier "incertitude",
   - si elle ne reconnaît pas → dossier "autre".

## Pourquoi l'IA peut se tromper (explication accessible)

- L'IA apprend à partir d'exemples. Si une espèce est rarement présente dans les exemples ou si la photo est mauvaise, l'IA peut mal interpréter.
- La confiance est une manière de dire "je suis assez sûr" ou "je ne suis pas sûr".

## Recherche visuelle — métaphore simple

Imagine que chaque photo reçoit une étiquette secrète composée de nombres. Ces nombres résument l'apparence de la photo (couleurs, formes, etc.).

On range ces étiquettes dans un annuaire spécial (l'index). Quand on a une nouvelle photo, on calcule aussi son étiquette et on demande à l'annuaire : "quelles photos ont des étiquettes proches ?". Les réponses sont les images qui ressemblent le plus.

Pourquoi utile : si l'IA doute, regarder les photos similaires peut nous aider à décider si la proposition est plausible.

## Aspects pratiques (pour l'utilisateur)

- Cooldown (délai entre captures) : empêche d'enchaîner trop vite les captures et protège l'ordinateur.
- Analyse en arrière-plan : l'ordinateur peut traiter la photo sans bloquer la vidéo, ce qui évite que la fenêtre se fige.

## Matériel et performances

- Sur un ordinateur ordinaire, tout fonctionne mais peut être un peu lent.
- Un ordinateur avec une carte graphique (GPU) est plus rapide pour l'IA.

## Confidentialité et bonnes pratiques

- Les images restent sur l'ordinateur. Faites attention avant de partager des photos contenant d'autres personnes.

## Glossaire très simple

- Frame : une photo prise depuis la caméra.
- Inférence : demander à l'IA d'analyser une photo.
- Vecteur/embedding : la petite étiquette de nombres qui décrit une image.
- Index : l'annuaire optimisé pour retrouver des images proches.

---

Souhaites-tu que je :

- rendre ce document plus court pour une notice terrain, ou
- l'illustrer avec schémas simples (capture → IA → stockage), ou
- le traduise en anglais ?

