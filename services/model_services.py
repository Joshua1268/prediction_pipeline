from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from datetime import datetime

from repositories.sales_data import SalesRepository
from services.models.neuralprophet import NeuralProphetForecast
from services.models.xgboost import XgboostPredictor
from shared.data_loader import DatabaseConnector
from transformations.sales_data_builder import SalesDataBuilder
from transformations.sales_data_preprocessor import SalesDataPreprocessor

import boto3
from dotenv import load_dotenv
import os

import warnings
warnings.filterwarnings("ignore", message="Importing plotly failed")

load_dotenv()


class SalesPredictionService:
    def __init__(
        self,
        s3_key_np: Optional[str] = None,
        s3_key_xgb: Optional[str] = None,
        s3_param_xgb:Optional[str] = None, 
        s3_param_np:Optional[str] = None, 
    ):
        try:
            self.repo =  SalesRepository()
            self.preprocessor =  SalesDataPreprocessor()
            self.neuralprophet = NeuralProphetForecast()
            self.xgboost = XgboostPredictor()
            self.connector = DatabaseConnector()
            

            self.s3_client = boto3.client('s3')
            self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
            self.s3_key_np = s3_key_np
            self.s3_key_xgb = s3_key_xgb
            self.s3_param_xgb = s3_param_xgb
            self.s3_param_np = s3_param_np
            
            if not self.bucket_name:
                print("Avertissement: AWS_S3_BUCKET_NAME n'est pas défini dans l'environnement.")

        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de SalesPredictionService: {e}")
            raise


    # --- Méthodes de chargement S3 pour les hyperparamètres ---

    def _load_s3_file(self, file_key: str) -> Dict[str, Any]:
        """Charge un fichier JSON (hyperparamètres ou features) depuis S3."""
        if not self.bucket_name:
            raise EnvironmentError("AWS_S3_BUCKET_NAME doit être défini pour charger les paramètres depuis S3.")
        
        try:
            print(f"Loading S3 file: s3://{self.bucket_name}/{file_key}")
            s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            file_content = s3_object['Body'].read().decode('utf-8')
            return json.loads(file_content)
        except self.s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Fichier de paramètres non trouvé sur S3: {file_key}")
        except Exception as e:
            raise IOError(f"Erreur lors du chargement du fichier S3 {file_key}: {e}")

    def _load_best_params(self, key_path: str) -> Dict[str, Any]:
        """Charge les hyperparamètres XGBoost depuis S3."""
        return self._load_s3_file(key_path)

    def _load_features(self, key_path: str) -> List[str]:
        """Charge les features depuis S3 et assure qu'ils sont une liste de strings."""
        data = self._load_s3_file(key_path)
        # Supposons que le fichier contient un dictionnaire avec la clé "features"
        if isinstance(data, dict) and "features" in data:
            data = data["features"]
            
        if not isinstance(data, list):
             raise ValueError(f"Le contenu du fichier de features S3 doit être une liste ou un dictionnaire avec la clé 'features'. Reçu: {type(data)}")
        return data

    def _load_neuralprophet_best_params(self, key_path: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
        """Charge les hyperparamètres et régresseurs NeuralProphet depuis S3."""
        
        params = self._load_s3_file(key_path)

        # On suppose que le fichier HPO contient une clé 'regressors' qui est une liste
        regressors = params.pop("regressors", [])
        params.pop("country_holidays", None)
        
        # Le retour doit être (best_params: Dict[str, Any], regressors: List[str]) pour NeuralProphet
        # La signature du train_neuralprophet utilise le type Dict[str, Dict[str, int]] pour regressors, 
        # je la maintiens pour la cohérence interne mais l'ajuste pour une liste si besoin.
        if isinstance(regressors, list):
             # On assume que la liste doit être convertie en Dict[str, Dict[str, int]] si nécessaire
             # Pour l'instant, je retourne la liste pour que la méthode train_neuralprophet gère l'adaptation.
             return params, regressors
        
        return params, regressors # Retourne params et regressors tels que trouvés
    
    # --- Méthodes de service (Data & HPO/Train) ---
    
    # Toutes les méthodes _load_raw_data, _build_dataset, et load_data restent inchangées 
    # car elles définissent le pipeline DB -> DF.

    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Logique de chargement des données brutes (inchangée)
        try :
            sales_df = self.repo.get_sales_data()
            discount_df = self.repo.get_discounts()
            customer_status_df = self.repo.get_customer_status()
            stockout_df = self.repo.get_stockouts()
            
            if sales_df.empty or "delivery_date" not in sales_df.columns:
                 raise ValueError("Les données de vente (sales_df) sont vides ou la colonne 'delivery_date' est manquante.")
                 
            holidays_df = self.repo.get_holidays(start_date=sales_df["delivery_date"].min())

            return (
                sales_df,
                discount_df,
                customer_status_df,
                stockout_df,
                holidays_df,
            )
        except Exception as e:
            raise Exception(f"❌ Échec du chargement des données brutes: {e}")
        finally:
            try:
                self.connector.close()
            except Exception as close_e:
                print(f"Avertissement: Échec de la fermeture du DatabaseConnector: {close_e}")


    def _build_dataset(
        self,
        sales_df: pd.DataFrame,
        discount_df: pd.DataFrame,
        customer_status_df: pd.DataFrame,
        stockout_df: pd.DataFrame,
        holidays_df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        try:
            builder = SalesDataBuilder(
                df_discount=discount_df,
                customer_status=customer_status_df,
                holidays=holidays_df,
                stockout=stockout_df,
            )
            return builder.transform(sales_df)
        except Exception as e:
            raise Exception(f"❌ Échec de la construction du dataset (SalesDataBuilder): {e}")


    def load_data(self) -> pd.DataFrame:
        try:
            print("🔌 Loading raw data from repository...")
            sales_df, discount_df, customer_status_df, stockout_df, holidays_df = (
                self._load_raw_data()
            )

            print("🧪 Building feature dataset...")
            transformed = self._build_dataset(
                sales_df=sales_df,
                discount_df=discount_df,
                customer_status_df=customer_status_df,
                stockout_df=stockout_df,
                holidays_df=holidays_df,
            )
            return transformed
        except Exception as e:
            print(f"❌ Échec de la procédure complète load_data: {e}")
            raise


    def hpo_xgboost(self, features: List[str]):
        try:
            df =  self.load_data()
            df = self.xgboost.preprocess(df)
            
            # Instanciation de OptunaXgboost ici (pas dans __init__)
            self.xgb_optuna = OptunaXgboost(df = df, features = features)
            # NOTE: La méthode optimize doit accepter les arguments Optuna (n_trials, storage_url, study_name)
            # Je suppose que votre script SageMaker les passe correctement ou que optimize a des valeurs par défaut.
            best_params , best_value = self.xgb_optuna.optimize()
            return best_params , best_value
        except Exception as e:
            print(f"❌ Échec de l'optimisation des hyperparamètres XGBoost (HPO): {e}")
            raise
        
    
    def train_xgboost(self) -> Path:
        # Logique d'entraînement (inchangée)
        try:
            df = self.load_data()
            print(f"👉 Loaded data shape: {df.shape}")

            df = self.xgboost.preprocess(df)

            print("⚙️ Loading model hyperparameters & features from S3...")
            features = self._load_features(self.s3_param_xgb)
            best_params = self._load_best_params(self.s3_param_xgb)
            
            if not hasattr(self, 'config') or not hasattr(self.config, 'xgboost_threshold'):
                raise AttributeError("La propriété 'self.config' ou 'self.config.xgboost_threshold' est manquante ou non initialisée.")

            print("🏃 Training XGBoost model...")
            
            X_train, y_train, X_test, y_test = self.xgboost.prepare_data(
                df=df,
                features=features,
                target='is_ordered',
            )

            model_path = self.xgboost.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=best_params,
                custom_threshold=self.config.xgboost_threshold
            )

            print(f"✅ XGBoost training completed. Model saved at: {model_path}")
            return Path(model_path)
        except Exception as e:
            print(f"❌ Échec de l'entraînement XGBoost: {e}")
            raise
    
    
    def train_neuralprophet(self) -> Path:
        # Logique d'entraînement (inchangée)
        try:
            df = self.load_data()

            print("🧹 Preprocessing (filtering, outliers, split)...")
            df = self.neuralprophet.process(df)
            print(f"👉 df: {df.shape}")

            print("⚙️ Loading model hyperparameters & regressors from S3...")
            features = self._load_features(self.s3_param_np)
            best_params, regressors = self._load_neuralprophet_best_params(self.s3_param_np)
            
            df = self.neuralprophet.prepare_data(df,features)

            print("🏃 Training NeuralProphet model...")
            model_path = self.neuralprophet.train_model(
                df=df,
                best_params=best_params,
                regressors=regressors,
            )

            print(f"✅ NeuralProphet training completed. Model saved at: {model_path}")
            return Path(model_path)
        except Exception as e:
            print(f"❌ Échec de l'entraînement NeuralProphet: {e}")
            raise

    # --- Méthodes de prédiction (inchangées) ---
    
    def predict_xgboost(
        self,
        start_date: Optional[datetime] = None,
        periods: int = 7,
        custom_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        # Logique de prédiction (inchangée)
        try:
            if start_date is None:
                start_date = datetime.now()
                
            if not hasattr(self, 'config') or not hasattr(self.config, 'xgboost_threshold'):
                raise AttributeError("La propriété 'self.config' ou 'self.config.xgboost_threshold' est manquante.")
            
            features = self._load_features(self.s3_param_xgb)
            print("⚙️ Generating features for XGBoost prediction...")
            
            df_to_predict_xgboost = self.xgboost.generate_features_for_prediction(
                start_date=start_date,
                periods=periods,
                sales_repo=self.repo
            )
            
            print("🔮 Generating XGBoost predictions...")
            threshold = custom_threshold if custom_threshold is not None else self.config.xgboost_threshold
            
            xgboost_predictions = self.xgboost.make_prediction(
                df_to_predict=df_to_predict_xgboost,
                features=features,
                custom_threshold=threshold,
                model_s3_uri=f"s3://{self.bucket_name}/{self.s3_key_xgb}"
            )
            
            return xgboost_predictions[xgboost_predictions['prediction_class'] == 1].copy()
        except Exception as e:
            print(f"❌ Échec de la prédiction XGBoost: {e}")
            return pd.DataFrame()


    def predict_neuralprophet(
        self,
        periods: int = 7
    ) -> pd.DataFrame:
        # Logique de prédiction (inchangée)
        try:
            features = self._load_features(self.s3_param_np)
        
            print("🔮 Generating NeuralProphet predictions...")
            
            return self.neuralprophet.generate_forecast(
                periods=periods, 
                features=features,
                model_s3_uri=f"s3://{self.bucket_name}/{self.s3_key_np}"
            ).copy()
        except Exception as e:
            print(f"❌ Échec de la prédiction NeuralProphet: {e}")
            return pd.DataFrame()


    def predict(
        self,
        start_date: Optional[datetime] = None,
        periods: int = 7,
        custom_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        # Logique de prédiction (inchangée)
        try:
            xgboost_results = self.predict_xgboost(
                start_date=start_date,
                periods=periods,
                custom_threshold=custom_threshold
            )
            
            neuralprophet_results = self.predict_neuralprophet(periods=periods)
            
            final_list = self._generate_final_purchase_list(
                xgboost_results,
                neuralprophet_results
            )

            print(f"✅ Final purchase list ready: {final_list.shape[0]} rows")
            return final_list
        except Exception as e:
            print(f"❌ Échec de la prédiction combinée (predict): {e}")
            return pd.DataFrame()


    def _generate_final_purchase_list(
        self,
        xgboost_predictions: pd.DataFrame,
        neuralprophet_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        # Logique de fusion (inchangée)
        try:
            xgboost_predictions['date'] = pd.to_datetime(xgboost_predictions['date'])
            neuralprophet_predictions['date'] = pd.to_datetime(neuralprophet_predictions['date'])

            print(f"XGBoost head:\n{xgboost_predictions.head()}")
            print(f"NeuralProphet head:\n{neuralprophet_predictions.head()}")

            
            final_predictions = pd.merge(
                xgboost_predictions,
                neuralprophet_predictions,
                on=['date', 'standard_name'],
                how='left'
            )
            
            return final_predictions.fillna({'predict': 0})
        except Exception as e:
            raise Exception(f"❌ Échec de la fusion des prédictions (final purchase list): {e}")