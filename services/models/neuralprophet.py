import os
import shutil  
from datetime import datetime
from typing import Dict, Optional, List

import pandas as pd
from neuralprophet import NeuralProphet as NP, save 

from transformations.sales_data_preprocessor import SalesDataPreprocessor

import boto3
from io import StringIO

import logging
import torch

# Configuration du logging pour NeuralProphet
logging.getLogger("NP").setLevel(logging.ERROR)


class NeuralProphetForecast(SalesDataPreprocessor): 
    
    def __init__(
        self,
        s3_bucket_name: Optional[str] = None, 
        s3_prefix: Optional[str] = None
    ):
        super().__init__()

        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.model = None


    def preprocess(
        self, raw_df: pd.DataFrame, min_days: int = 110
    ) -> pd.DataFrame:
        
        df = raw_df.copy()
        
        try:
            df["delivery_date"] = pd.to_datetime(df["delivery_date"])

            df = self._filter_by_min_sales_days(df, min_days=min_days)
            df = self._filter_by_recurrence(df)

            train_df, test_df = self._split_train_test(df)
            clean_train_df = self._remove_outliers(train_df)
            df = pd.concat([clean_train_df, test_df])
            
            return df
        except Exception as e:
            print(f"❌ Erreur lors du traitement initial des données: {e}")
            return pd.DataFrame()
    

    def prepare_and_upload_data(
        self, df: pd.DataFrame, 
    ) -> str:
       
        if not self.s3_bucket_name or not self.s3_prefix:
            raise ValueError("Le nom du bucket S3 et le préfixe doivent être définis pour l'upload.")
        
        print(df.isnull().sum())
        print("Préparation des données pour l'upload Sagemaker (NeuralProphet)...")
        
        try:
            df.rename(
                columns={"delivery_date": "ds", "quantity": "y", "standard_name": "ID"},
                inplace=True,
            )
            
            # Préparation du CSV en mémoire
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Définition de la clé S3
            timestamp_str = datetime.now().strftime('%Y%m%d')
            file_key = f"{self.s3_prefix}/training_data_{timestamp_str}.csv"
            s3_uri = f"s3://{self.s3_bucket_name}/{file_key}"
            
            print(f"Uploading data to S3: {s3_uri}")
            
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=self.s3_bucket_name,
                Key=file_key,
                Body=csv_buffer.getvalue()
            )
            print("✅ Data uploaded successfully.")
            
            return s3_uri
        except Exception as e:
            print(f"❌ Échec de l'upload des données vers S3: {e}")
            raise


    
    def generate_forecast(
        self, 
        periods: int = 3, 
        features: Optional[List[str]] = None, 
        model_s3_uri: Optional[str] = None
    ) -> pd.DataFrame:
        
        if not (1 <= periods <= 7):
            raise ValueError("La période ('periods') doit être entre 1 et 7 jours.")
        
        temp_file_path = None
        
        try:
            # 1. Chargement du modèle si nécessaire
            if self.model is None:
                if model_s3_uri:
                    print(f"Loading model from S3 URI: {model_s3_uri}")
                    
                    s3 = boto3.client('s3')
                    # Extraction du bucket et de la clé
                    bucket_name, key = model_s3_uri.replace("s3://", "").split("/", 1)
                    
                    # Chemin du fichier temporaire
                    temp_file_path = os.path.join(self.cache_dir, "temp_neuralprophet.np")
                    
                    # Téléchargement et chargement du modèle
                    s3.download_file(bucket_name, key, temp_file_path)
                    
                    # NOTE: weights_only=False est crucial pour NeuralProphet
                    self.model = torch.load(temp_file_path, weights_only=False)
                    self.model.restore_trainer()
                    
                    print("✅ Modèle téléchargé et chargé avec succès depuis S3.")
                else:
                    raise RuntimeError("L'URI S3 du modèle est requis pour la prédiction si le modèle n'est pas préchargé.")

            # 2. Données PLACHOLDER (le DataFrame historique doit être utilisé pour make_future_dataframe)
            # NOTE: La logique originale utilise un DataFrame vide, ce qui ne permet pas une prédiction réelle
            # avec make_future_dataframe() basé sur des IDs existants. On conserve la structure pour ne pas
            # modifier la LOGIQUE.
            df = pd.DataFrame(columns=['ds', 'y', 'ID']) 
            print("Note: Les données historiques doivent être passées en argument pour la prévision.")
            
            # 3. Génération du DataFrame futur
            future = self.model.make_future_dataframe(
                df, 
                n_historic_predictions=True, 
                periods=periods
            )
            
            # 4. Réalisation des prédictions
            forecast = self.model.predict(future)

            # 5. Extraction et formatage des prévisions
            last_historic_date = df["ds"].max()
            # On cherche à isoler la ligne correspondant à la fin de l'historique pour extraire les yhatN
            last_historic_row = forecast[forecast['ds'] == last_historic_date].copy()
            
            if last_historic_row.empty:
                raise ValueError("Impossible de trouver la dernière date historique dans le DataFrame de prévision.")

            id_vars = ['ID', 'ds']
            yhat_cols = [f"yhat{i}" for i in range(1, periods + 1)]

            if not all(col in last_historic_row.columns for col in yhat_cols):
                raise KeyError(f"Colonne(s) yhat manquante(s) dans la dernière ligne historique: {yhat_cols}")

            forecast_long = pd.melt(
                last_historic_row,
                id_vars=id_vars,
                value_vars=yhat_cols,
                var_name='period_offset',
                value_name='predict'
            )
            
            # Calculer les dates de prévision futures
            forecast_long['period_offset'] = forecast_long['period_offset'].str.extract('(\d+)').astype(int)
            forecast_long['date'] = forecast_long['ds'] + pd.to_timedelta(forecast_long['period_offset'], unit='D')
            
            # Filtrer pour ne garder que les prévisions futures réelles
            forecast_long = forecast_long[(forecast_long['date'] > last_historic_date) & 
                                        (forecast_long['date'] <= last_historic_date + pd.Timedelta(days=periods))]
            
            # Renommer et nettoyer le DataFrame final
            forecast_long.rename(columns={'ID': 'standard_name'}, inplace=True)
            forecast_long = forecast_long[['standard_name', 'date', 'predict']]
            forecast_long['date'] = forecast_long['date'].dt.date

            # Clip les valeurs négatives à zéro
            forecast_long['predict'] = forecast_long['predict'].clip(lower=0)

            print(f"Prévision générée: du {forecast_long['date'].min()} au {forecast_long['date'].max()}")

            return forecast_long
            
        except RuntimeError as re:
            raise re
        except Exception as e:
            print(f"❌ Erreur lors de la génération de la prévision: {e}")
            raise
        finally:
            # Nettoyage : suppression du fichier modèle temporaire s'il a été téléchargé
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    print(f"Nettoyage: Fichier temporaire supprimé: {temp_file_path}")
                except Exception as e_clean:
                    print(f"Avertissement: Impossible de supprimer le fichier temporaire {temp_file_path}: {e_clean}")
            
            # Aucune self.database_connector.close() n'est nécessaire ici.