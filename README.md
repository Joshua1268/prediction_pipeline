# Pipeline de Prévision avec Interface Streamlit

Ce projet fournit une solution complète pour générer des données synthétiques de vente, entraîner automatiquement des modèles prédictifs et offrir une interface Streamlit interactive pour les prédictions de produits et légumes.

## Fonctionnalités

- Génération automatique de données synthétiques pour les ventes, l'inventaire et le comportement client
- Entraînement automatique du modèle avec NeuralProphet pour la prévision de séries chronologiques
- Interface web Streamlit interactive pour les prédictions et requêtes
- Capacité d'exécution locale

## Prérequis

- Python 3.11
- Packages requis listés dans requirements.txt

## Installation

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

2. Exécutez l'application principale :
```bash
streamlit run app.py
```

## Utilisation

L'application va :
1. Générer des données synthétiques si elles ne sont pas déjà présentes
2. Entraîner automatiquement le modèle de prédiction
3. Lancer l'interface Streamlit où vous pouvez :
   - Visualiser les tendances historiques des données
   - Faire des prévisions futures pour les produits et légumes
   - Poser des questions sur les prédictions
   - Visualiser les résultats des prévisions

## Architecture

- `generate_synthetic_data.py`: Crée des données de vente et d'inventaire réalistes
- `model_trainer.py`: Gère l'entraînement automatique du modèle
- `app.py`: Interface Streamlit pour l'interaction utilisateur
- `predictor.py`: Logique de prédiction principale