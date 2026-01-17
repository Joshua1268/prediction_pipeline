import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import pickle
import os
from datetime import datetime, timedelta
import holidays
from typing import Dict, List, Tuple, Optional
from generate_synthetic_data import SimpleSyntheticDataGenerator


class ProductPredictionModel:
    def __init__(self):
        self.model = None
        self.data = None
        self.forecast_data = None

    def load_or_generate_data(self, data_path: str = "data/synthetic_sales_data.csv"):
        """Charge les données existantes ou en génère de nouvelles"""
        if os.path.exists(data_path):
            print(f"Chargement des données existantes depuis {data_path}")
            self.data = pd.read_csv(data_path)
        else:
            print("Génération de nouvelles données synthétiques...")
            generator = SimpleSyntheticDataGenerator(start_date="2022-01-01", end_date="2024-12-31")
            sales_data, _, _ = generator.generate_all_data()

            # Créer le répertoire de données s'il n'existe pas
            os.makedirs("data", exist_ok=True)

            # Sauvegarder les données
            df = pd.DataFrame(sales_data)
            df.to_csv(data_path, index=False)
            self.data = df

        # Préparer les données pour NeuralProphet - garder seulement ds et y
        self.data = self.data[['date', 'quantity_sold']].copy()
        self.data.rename(columns={'date': 'ds', 'quantity_sold': 'y'}, inplace=True)
        self.data['ds'] = pd.to_datetime(self.data['ds'])

        # Supprimer les duplicatas de dates
        self.data = self.data.drop_duplicates(subset=['ds'])

        # Trier par date
        self.data = self.data.sort_values('ds').reset_index(drop=True)

        return self.data

    def train_model(self, data: pd.DataFrame = None):
        """Entraîne le modèle de prévision"""
        if data is not None:
            self.data = data

        if self.data is None or self.data.empty:
            raise ValueError("Aucune donnée disponible pour l'entraînement")

        # Initialiser le modèle NeuralProphet
        self.model = NeuralProphet(
            n_forecasts=30,
            n_lags=7,
            changepoints_range=0.95,
            n_changepoints=15,
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=True,
            epochs=50,
            quantiles=[0.1, 0.9]  # Activer les intervalles de confiance
        )

        # Entraîner le modèle
        metrics = self.model.fit(self.data, freq='D')

        print("Modèle entraîné avec succès!")
        return metrics

    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Effectue une prédiction pour les périodes spécifiées"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné")

        # Créer des dates futures
        future = self.model.make_future_dataframe(self.data, periods=periods)

        # Faire la prédiction
        forecast = self.model.predict(future)
        self.forecast_data = forecast

        return forecast

    def get_prediction_for_product(self, product_name: str, periods: int = 30) -> Dict:
        """Obtenir une prédiction spécifique pour un produit"""
        # Pour cette implémentation simplifiée, nous utilisons les données générales
        # Dans une version plus avancée, on pourrait filtrer par produit
        forecast = self.predict(periods)

        # Extraire les prédictions futures
        future_predictions = forecast.tail(periods)

        result = {
            'product': product_name,
            'predictions': future_predictions[['ds', 'yhat1']].to_dict('records')
        }

        # Vérifier si les intervalles de confiance existent
        if 'yhat_lower' in future_predictions.columns:
            result['lower_bound'] = future_predictions[['ds', 'yhat_lower']].to_dict('records')
        else:
            result['lower_bound'] = [{'ds': row['ds'], 'yhat_lower': row['yhat1']} for row in future_predictions[['ds', 'yhat1']].to_dict('records')]

        if 'yhat_upper' in future_predictions.columns:
            result['upper_bound'] = future_predictions[['ds', 'yhat_upper']].to_dict('records')
        else:
            result['upper_bound'] = [{'ds': row['ds'], 'yhat_upper': row['yhat1']} for row in future_predictions[['ds', 'yhat1']].to_dict('records')]

        return result

    def get_insights(self) -> Dict:
        """Fournit des informations sur les prédictions"""
        if self.forecast_data is None:
            self.predict()

        latest_actual = self.data.tail(7)['y'].mean() if len(self.data) >= 7 else self.data['y'].mean()
        latest_forecast = self.forecast_data.tail(7)['yhat1'].mean()

        trend = "hausse" if latest_forecast > latest_actual else "baisse"
        difference = abs(latest_forecast - latest_actual)

        insights = {
            'trend': trend,
            'difference': difference,
            'latest_average_actual': latest_actual,
            'latest_average_forecast': latest_forecast
        }

        return insights