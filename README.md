# Plateforme d’Analyse Prédictive Fruits & Légumes

**Auteur : Josué Kouassi**
**Data Scientist & ML Ops Engineer**

## Présentation générale

Cette plateforme a été conçue et développée par **Josué Kouassi, Data Scientist & ML Ops Engineer**, dans une démarche professionnelle visant à démontrer des compétences avancées en analyse prédictive, modélisation statistique et déploiement de solutions data-driven appliquées aux produits périssables.

La **Plateforme d’Analyse Prédictive Fruits & Légumes** est un outil basé sur l’IA qui permet de prévoir la demande, d’optimiser les stocks et de gérer les produits périssables (fruits et légumes). Ce système offre une solution complète aux acteurs du secteur agricole et agroalimentaire afin de prendre des décisions éclairées en matière de gestion des stocks et de stratégies de prix.

## Fonctionnalités

* **Vue d’ensemble du tableau de bord** : Visualisation des indicateurs clés et des tendances des données fruits et légumes
* **Exploration des données** : Analyse interactive des ventes, des stocks et des données clients avec des insights saisonniers
* **Prévision Fruits & Légumes** : Anticipation de la demande future par type de fruit ou de légume
* **Planification des stocks** : Optimisation des niveaux de stock et suggestion de points de réapprovisionnement en tenant compte de la périssabilité
* **Optimisation des prix** : Stratégies de tarification dynamique basées sur les prévisions de demande et la durée de conservation

## Structure du projet

```
fv_predictive_analytics_platform/
├── app.py                   
├── generate_synthetic_data.py 
├── prediction_model.py      
├── requirements.txt          
├── data/                     
│   ├── fv_sales_data.csv
│   ├── fv_inventory_data.csv
│   └── fv_customer_data.csv
└── README.md                 
```

## Prérequis

* Python 3.8 ou version ultérieure
* Gestionnaire de paquets pip

## Installation

1. Clonez ou téléchargez ce dépôt sur votre machine locale.

2. Accédez au répertoire du projet :

```bash
cd fv_predictive_analytics_platform
```

3. Créez un environnement virtuel (recommandé) :

```bash
python3 -m venv venv
source venv/bin/activate  # Sous Windows : venv\Scripts\activate
```

4. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer l’application

1. Depuis le répertoire du projet, lancez l’application Streamlit :

```bash
streamlit run app.py
```

2. L’application s’ouvrira dans votre navigateur par défaut à l’adresse `http://localhost:8501`

3. Si le navigateur ne s’ouvre pas automatiquement, accédez manuellement à l’URL affichée.

### Sections de l’application

* **Tableau de bord** : Présente les indicateurs clés et visualisations des données fruits et légumes
* **Exploration des données** : Permet de filtrer et d’analyser les données historiques avec une vision saisonnière
* **Prévisions Fruits & Légumes** : Génère des prévisions de demande pour des produits spécifiques
* **Planification des stocks** : Fournit des recommandations d’optimisation des stocks en tenant compte de la durée de conservation
* **Optimisation des prix** : Propose des stratégies de tarification dynamique basées sur les prévisions de demande

### Génération des données

L’application génère automatiquement des données synthétiques de fruits et légumes lors du premier lancement. Pour régénérer les données manuellement :

```bash
python generate_synthetic_data.py
```

## Architecture des modèles

La plateforme se compose de :

1. **Générateur de données Fruits & Légumes** : Crée des données réalistes de ventes, de stocks et de clients intégrant des effets saisonniers
2. **Modèle de prédiction spécialisé** : Utilise des modèles comme Random Forest ou Régression Linéaire pour prédire la demande en intégrant la saisonnalité et la périssabilité
3. **Modèle de prévision** : Étend les prédictions sur des périodes futures avec des intervalles de confiance adaptés aux produits périssables
4. **Module de planification des stocks** : Calcule les points de réapprovisionnement et quantités optimales en tenant compte de la durée de vie des produits
5. **Moteur d’optimisation des prix** : Suggère des prix dynamiques basés sur la demande prévue et la périssabilité

## Dépannage

* En cas de problème de dépendances, essayez de mettre à jour pip : `pip install --upgrade pip`
* Si Streamlit ne démarre pas, vérifiez que toutes les dépendances sont installées : `pip install -r requirements.txt`
* En cas de problèmes de performance avec de grands volumes de données, envisagez un échantillonnage

## Technologies utilisées

* **Python** : Langage de programmation principal
* **Streamlit** : Framework d’application web
* **Plotly** : Visualisations interactives
* **Pandas** : Manipulation des données
* **Scikit-learn** : Algorithmes de machine learning
* **NumPy** : Calculs numériques

## À propos de la plateforme

La Plateforme d’Analyse Prédictive Fruits & Légumes répond aux principaux enjeux de la gestion des produits périssables :

1. **Prévision de la demande** : Anticipation de la demande future avec ajustements saisonniers
2. **Optimisation des stocks** : Prise en compte de la durée de conservation et de la périssabilité
3. **Tarification dynamique** : Ajustement des prix en fonction des prévisions de demande et de la durée de vie des produits
4. **Interface utilisateur** : Interface web basée sur Streamlit pour une utilisation simple
5. **Analyse saisonnière** : Intégration des schémas saisonniers de consommation des fruits et légumes

## Auteur & Crédits

**Josué Kouassi**
*Data Scientist & ML Ops Engineer*

Spécialisé en analyse de données, machine learning et industrialisation de modèles (MLOps), avec un intérêt particulier pour les problématiques métier liées à la supply chain, à la prévision de la demande et à l’optimisation des performances opérationnelles.

