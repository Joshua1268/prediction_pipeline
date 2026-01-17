import os
import json
from datetime import datetime, timedelta
from typing import Tuple, Optional, List

import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from transformations.sales_data_preprocessor import (
    SalesDataPreprocessor,
)
from shared.data_loader import DatabaseConnector
from repositories.sales_data import SalesRepository

import boto3
from io import StringIO


class XgboostPredictor(SalesDataPreprocessor):

    def __init__(
        self,
        s3_bucket_name: Optional[str] = None, 
        s3_prefix: Optional[str] = None
    ):
        self.sales_data_repository = SalesRepository()
        self.database_connector = DatabaseConnector()
        self.s3_bucket_name = s3_bucket_name
        self.s3_prefix = s3_prefix
        self.model = None

    def preprocess(self, raw_df: pd.DataFrame, min_days: int = 110) -> pd.DataFrame:
       
        df = raw_df.copy()
        
        try:
            df["delivery_date"] = pd.to_datetime(df["delivery_date"])
            
            df = self._add_lag(df, 'active', 1)
            df = self._add_lag(df, 'unit_price', 7)
            df = self._add_lag(df, 'stockout', 7)
            
            df = self.fillna_columns(['unit_price_lag7', 'stockout_lag7'], df)
            
            df['is_ordered'] = (df['count_orderId'] > 0).astype(int) 
            
            return df
        except Exception as e:
            print(f"❌ Erreur lors du prétraitement des données: {e}")
            return pd.DataFrame()


    def prepare_and_upload_data(
        self,
        df: pd.DataFrame,
        target: str = 'is_ordered',
    ) -> str:
        
        if not self.s3_bucket_name or not self.s3_prefix:
            raise ValueError("Le nom du bucket S3 et le préfixe doivent être définis pour l'upload.")
            
        print("Préparation des données pour l'upload Sagemaker...")
        
        try:
            print(df.isnull().sum())
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            
            if dropped_rows > 0:
                print(f"Attention: {dropped_rows} lignes avec des valeurs NaN ont été supprimées.")
                
            if df.empty:
                raise RuntimeError("Le DataFrame final est vide après nettoyage des NaN.")

            print(f"Data final shape: {df.shape}")

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
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
            
        except RuntimeError as re:
            raise re
        except Exception as e:
            print(f"❌ Échec de l'upload des données vers S3: {e}")
            raise 


    def generate_features_for_prediction(
        self, 
        start_date: datetime, 
        periods: int = 1, 
        sales_repo: Optional[SalesRepository] = None
    ) -> pd.DataFrame:
       
        if periods > 7:
            raise ValueError("Le nombre de périodes ne peut excéder 7 jours.")
        
        if sales_repo is None:
            print("Avertissement: L'instance SalesRepository fournie est ignorée, self.sales_data_repository est utilisé.")
        
        connection = None 

        try:
            # 1. Chargement des données de base et préparation du DataFrame de base
            date_range = [start_date + timedelta(days=i) for i in range(periods)]
            holidays_start_date = (start_date - timedelta(days=7)).strftime('%Y-%m-%d')
            
            customer_status = self.sales_data_repository.get_customer_status()
            holidays_df = self.sales_data_repository.get_holidays(start_date=holidays_start_date)
            
            df_range = pd.DataFrame({'delivery_date': date_range})
            df_range['delivery_date'] = pd.to_datetime(df_range['delivery_date'])
            
            holidays_df['delivery_date'] = pd.to_datetime(holidays_df['delivery_date'])
            df_range = pd.merge(df_range, holidays_df, on='delivery_date', how='left')
            df_range['is_holiday'] = df_range['is_holiday'].fillna(0).astype(int)

            # 2. Récupération des standards de produits actifs
            connection = self.database_connector.connect()
            query_standards = """
                SELECT
                    ps.id AS standard_id,
                    ps.name AS standard_name,
                    ut2.title AS default_unit
                FROM product_standards ps
                JOIN products p ON ps.id = p.product_standard_id
                JOIN unit_translations ut2 ON ut2.unit_id = ps.unit_id
                WHERE ps.deleted_at IS NULL
                AND p.deleted_at IS NULL
                AND ut2.deleted_at IS NULL
                AND p.category_id NOT IN (8, 10, 11)
            """
            list_standards_ids = pd.read_sql(query_standards, connection)

            # 3. Création des lignes de caractéristiques basées sur la date
            data_frames = []
            for index, row in df_range.iterrows():
                day_df = list_standards_ids.copy()
                day_df['delivery_date'] = row['delivery_date']
                day_df['day_of_week'] = row['delivery_date'].weekday() + 1
                day_df['week_of_month'] = (row['delivery_date'].day - 1) // 7 + 1
                day_df['month_of_year'] = row['delivery_date'].month
                day_df['day'] = row['delivery_date'].day
                day_df['week'] = row['delivery_date'].isocalendar().week
                day_df['is_holiday'] = row['is_holiday']
                data_frames.append(day_df)
                
            data = pd.concat(data_frames, ignore_index=True)

            # 4. Ajout du lag 'active' (statut client de la semaine précédente)
            last_week_start_dt = start_date - timedelta(days=start_date.weekday() + 7)
            last_week_start_str = last_week_start_dt.strftime('%Y-%m-%d')
            
            customer_status['week_start'] = pd.to_datetime(customer_status['week_start'])
            last_customer_status = customer_status[
                customer_status['week_start'] == pd.to_datetime(last_week_start_str)
            ]
            pivot_status = last_customer_status.pivot_table(
                index='week_start', columns='status', values='shop_count', fill_value=0
            ).reset_index()
            pivot_status = pivot_status.rename(columns={'active': 'active_lag1'})

            data['week_start'] = pd.to_datetime(last_week_start_dt)
            pivot_status['week_start'] = pd.to_datetime(pivot_status['week_start'])
            data = pd.merge(data, pivot_status, on='week_start', how='left')
            data.drop(columns=['week_start'], inplace=True) 

            # 5. Ajout du lag 'unit_price' (prix moyen de la dernière livraison)
            last_week_day = start_date - timedelta(days=1)
            
            unit_price_lag_query = """
                WITH last_prices AS (
                SELECT
                    ps.id AS standard_id,
                    MAX(o.delivery_date) AS last_delivery_date
                FROM orders o
                JOIN order_details od ON o.id = od.order_id
                JOIN stocks s ON s.id = od.stock_id
                JOIN products p ON p.id = s.countable_id
                JOIN product_standards ps ON ps.id = p.product_standard_id 
                WHERE o.shop_id <> 43
                AND o.status NOT IN ('canceled','proforma')
                AND o.delivery_date <= %(last_week_day)s
                GROUP BY ps.id
            )
            SELECT
                lp.last_delivery_date,
                ps.id AS standard_id,
                AVG(od.unit_price / p.weight) AS unit_price_lag7
            FROM last_prices lp
            JOIN orders o ON o.delivery_date = lp.last_delivery_date
            JOIN order_details od ON o.id = od.order_id
            JOIN stocks s ON s.id = od.stock_id
            JOIN products p ON p.id = s.countable_id
            JOIN product_standards ps ON ps.id = p.product_standard_id 
            WHERE ps.id = lp.standard_id
            AND o.shop_id <> 43
            AND o.status NOT IN ('canceled','proforma')
            GROUP BY ps.id, lp.last_delivery_date
            """
            unit_price_lag = pd.read_sql_query(
                unit_price_lag_query, 
                connection, 
                params={'last_week_day': last_week_day.strftime('%Y-%m-%d')}
            )
            data = pd.merge(data, unit_price_lag[['standard_id', 'unit_price_lag7']], on='standard_id', how='left')

            # 6. Ajout du lag 'stockout' (nombre de commandes en rupture la veille)
            stockout_query = """ 
                SELECT
                    ps.id AS standard_id,
                    COUNT(DISTINCT od.order_id) AS stockout_lag7
                FROM orders o 
                JOIN order_details od ON o.id = od.order_id
                JOIN stocks s ON s.id = od.stock_id
                JOIN products p ON p.id = s.countable_id
                JOIN product_standards ps ON ps.id = p.product_standard_id 
                WHERE od.reason_delete ='rupture' AND o.delivery_date = %(last_week_day)s
                GROUP BY ps.id, o.delivery_date
            """
            stockout_lag7 = pd.read_sql_query(
                stockout_query, 
                connection, 
                params={'last_week_day': last_week_day.strftime('%Y-%m-%d')}
            )

            if stockout_lag7.empty:
                data['stockout_lag7'] = 0
            else:
                data = pd.merge(data, stockout_lag7[['standard_id', 'stockout_lag7']], on='standard_id', how='left')
                data['stockout_lag7'] = data['stockout_lag7'].fillna(0)
            
            return data
            
        except Exception as e:
            print(f"❌ Erreur lors de la génération des features pour la prédiction: {e}")
            raise 
        finally:
            # FERMETURE DE LA CONNEXION À LA BASE DE DONNÉES
            if connection:
                self.database_connector.close()


    def make_prediction(
        self,
        df_to_predict: pd.DataFrame,
        features: List[str],
        custom_threshold: float = 0.4,
        model_s3_uri: Optional[str] = None
    ) -> pd.DataFrame:
        
        try:
            # 1. Chargement du modèle si nécessaire
            if self.model is None:
                if model_s3_uri:
                    print(f"Loading model from S3 URI: {model_s3_uri}")
                    s3 = boto3.client('s3')
                    
                    self.model = xgb.XGBClassifier()
                    # Simuler le chargement et lever l'erreur si la logique réelle est requise
                    raise NotImplementedError("Le chargement réel du modèle depuis S3 n'est pas implémenté dans cette simulation.")
                    
                    print("✅ Model loaded successfully from S3 (simulated).")
                else:
                    raise RuntimeError("L'URI S3 du modèle est requis pour la prédiction si le modèle n'est pas préchargé.")

            # 2. Vérification des caractéristiques
            missing_cols = set(features) - set(df_to_predict.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans l'input pour la prédiction: {missing_cols}")

            # 3. Prédiction
            X_predict = df_to_predict[features]
            probabilities = self.model.predict_proba(X_predict)[:, 1]
            
            df_to_predict['prediction_prob'] = probabilities
            df_to_predict['prediction_class'] = (probabilities >= custom_threshold).astype(int)

            print(f"Prédictions générées pour {len(df_to_predict)} lignes.")

            # 4. Nettoyage et formatage du résultat
            df_to_predict = df_to_predict.rename(columns={'delivery_date': 'date'})
            df_to_predict['date'] = df_to_predict['date'].dt.date
            
            cols_to_keep = ['standard_id', 'standard_name', 'default_unit', 'date', 'prediction_class']
            df_to_predict = df_to_predict[cols_to_keep].drop_duplicates().reset_index(drop=True)

            print('Colonnes finales formatées pour le résultat.')

            return df_to_predict
            
        except Exception as e:
            print(f"❌ Erreur lors de l'étape de prédiction: {e}")
            return pd.DataFrame(columns=['standard_id', 'standard_name', 'default_unit', 'date', 'prediction_class'])