import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
import pickle
import os
from datetime import datetime, timedelta
from prediction_model import ProductPredictionModel


class ModelTrainer:
    def __init__(self):
        self.predictor = ProductPredictionModel()
        self.model_path = "models/trained_model.pkl"
        
    def load_or_train_model(self):
        """Charge un modèle existant ou en entraîne un nouveau"""
        # Créer le répertoire des modèles s'il n'existe pas
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(self.model_path):
            print("Chargement du modèle existant...")
            with open(self.model_path, 'rb') as f:
                self.predictor = pickle.load(f)
        else:
            print("Entraînement d'un nouveau modèle...")
            data = self.predictor.load_or_generate_data()
            self.predictor.train_model(data)
            
            # Sauvegarder le modèle entraîné
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.predictor, f)
                
        return self.predictor
    
    def retrain_model(self):
        """Réentraîne le modèle avec les dernières données"""
        print("Réentraînement du modèle...")
        data = self.predictor.load_or_generate_data()
        self.predictor.train_model(data)
        
        # Sauvegarder le modèle mis à jour
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.predictor, f)
            
        return self.predictor