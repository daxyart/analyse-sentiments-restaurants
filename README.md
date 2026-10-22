# Analyse de sentiment d’avis de restaurant

Petit projet d’apprentissage automatique en Python qui classe des avis en français comme **positifs** ou **négatifs**.

## Fonctionnement

Le programme lit `avis_restaurants.csv`, transforme les mots et expressions en caractéristiques avec `CountVectorizer`, puis utilise `MultinomialNB` pour prédire le sentiment. Les deux étapes sont réunies dans un pipeline scikit-learn.

Le jeu de données contient 24 exemples pédagogiques. Il est séparé en données d’entraînement et de test (75 % / 25 %). Le résultat observé est **67 % de bonnes réponses sur 6 avis de test** (4 sur 6). Ce score est indicatif : le jeu de données est très petit et les exemples ne sont pas un échantillon représentatif d’avis réels du restaurant.

Après cette évaluation, le modèle est réentraîné sur les 24 avis pour permettre les essais interactifs.

## Installation et lancement (Windows)

Dans le dossier du projet, ouvre un terminal et exécute :

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe classificateur_restaurants.py
```

Saisis un avis lorsque le programme le demande. Tape `quitter` pour arrêter.

## Fichiers

- `classificateur_restaurants.py` : chargement des données, entraînement, évaluation et prédiction interactive.
- `avis_restaurants.csv` : exemples d’avis français avec leur étiquette (`positif` ou `négatif`).
- `requirements.txt` : dépendance Python du projet.
