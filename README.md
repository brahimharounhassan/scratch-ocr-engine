## Nom

Scratch OCR Engine
"Un moteur OCR personnalisé pour détecter, analyser et convertir des algorithmes Scratch à partir d'images en code Python exécutable."

## Description

Scratch OCR Engine est un outil développé pour analyser des images contenant des algorithmes visuels Scratch, détecter les blocs logiques, d’extraire le texte associé, puis de générer un graphe représentant l’algorithme, et finalement le convertir en code Python exécutable.

Ce projet a été réalisé dans le cadre :

    D’un stage à l’Unité de Recherche RTIM  (Gabès, Tunisie) et à Intellect Academy

## Technologies & Frameworks

## Fonctionnalités Principales

- Prétraitement des images : Amélioration de la qualité d’image pour faciliter l’analyse
- Segmentation des blocs Scratch : Identification des blocs logiques par contour et Deep Learning
- Détection et extraction du texte : Extraction du contenu textuel à l’intérieur des blocs
- Génération d’un graphe orienté : Modélisation de la logique avec un algorithme DFS
- Conversion en code Python : Transformation du graphe en script Python exécutable

## Structure du Projet

- [x] `scratch-ocr-engine/` :
      [x] `DL_implementation` :
      [x] `classes` :
      [x] `images` :
      [x] `maskrcnnp` :
      [x] `modules` :
      [x] `pythonCode` :
      [x] `scripts` :
      [x] `README.md`

## Installation

1. Cloner le dépôt :
   ```
   git clone https://github.com/brahimharounhassan/scratch-ocr-engine.git
   cd scratch-ocr-engine
   ```
2. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```
3. Télécharger les modèles nécessaires :
   - frozen_east_text_detection.pb (modèle EAST)
   - Modèles Mask R-CNN pré-entraînés

## Exemple d’utilisation

    ```
    python scripts/inputs.py
    ```
    Le programme traitera l’image dans le répertoire `images/`, détectera les blocs, extraira le texte, générera un graphe et produira un fichier Python correspondant.

## Références

- [EAST: Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Graphviz](https://graphviz.org/)

## Auteur

- Haroun Hassan BRAHIM - Ingénieur en Intelligence Artificielle | Spécialisé en Machine Learning et Deep Learning
  [LinkedIn](www.linkedin.com/in/brahimharounhassan/) | [Github](https://github.com/brahimharounhassan) | [ResearchGate](https://www.researchgate.net/profile/Haroun-Hassan-Brahim)

- Mahamat Issa CHOUEB

## Licence

Ce projet est sous licence MIT
