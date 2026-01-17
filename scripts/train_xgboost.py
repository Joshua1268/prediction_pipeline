import argparse
import json
import pandas as pd
from pathlib import Path
import os
import mlflow
import boto3
from io import BytesIO
from typing import Dict, Any, Union
# Import des classes de service
from services.model_services import SalesPredictionService
from services.mlflow_manager import MLflowManager
# Import des classes de modèle (pour les types et la logique)
from services.models.xgboost import XgboostPredictor 
import xgboost as xgb # Nécessaire pour le logging MLflow


# --- UTILS S3 (Pour le chargement direct en DataFrame) ---
def s3_csv_to_df(bucket_name: str, key: str, **kwargs) -> pd.DataFrame:
    """ Télécharge un fichier CSV depuis S3 et le charge dans un DataFrame. """
    try:
        s3 = boto3.client('s3')
        print(f"Téléchargement des données S3: s3://{bucket_name}/{key}")
        
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        data = obj['Body'].read()
        
        # Utiliser BytesIO pour charger directement en mémoire
        df = pd.read_csv(BytesIO(data), **kwargs)
        
        # S'assurer que la colonne cible est présente
        if 'is_ordered' not in df.columns:
            raise ValueError("La colonne cible 'is_ordered' est manquante dans les données S3.")
        
        return df
    except Exception as e:
        raise IOError(f"❌ Erreur de chargement du fichier S3 {key}: {e}")

def train_xgboost_final(args: Dict[str, Any]):
    """ Entraîne le modèle final XGBoost en utilisant les paramètres HPO et logue via MLflow. """
    
    mlflow_manager = MLflowManager()
    mlflow_manager.setup()

    # 1. Configuration de l'environnement S3 et des chemins
    try:
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        if not bucket_name:
            raise EnvironmentError("La variable d'environnement 'AWS_S3_BUCKET_NAME' doit être définie.")
            
        params_key = args.get("s3_params_key")
        data_key = args.get("s3_data_key")
        model_output_dir = args.get("sm_model_dir", "/opt/ml/model") # Répertoire de sortie SageMaker
        
        # Initialisation du SalesPredictionService (pour l'accès S3 aux paramètres et les objets modèles)
        prediction_service = SalesPredictionService(
            s3_param_xgb=params_key 
        )
        xgboost_model_logic = prediction_service.xgboost
        target_column = 'is_ordered'

    except Exception as e:
        print(f"❌ Échec de la configuration: {e}")
        raise
        
    # --- Démarrage du RUN MLFLOW ---
    with mlflow.start_run(run_name="XGBoost_Final_Model"):
        try:
            print("--- Démarrage de l'entraînement final XGBoost (S3 Data & MLflow) ---")
            
            # 2. Lecture du fichier de meilleurs paramètres HPO depuis S3
            print(f"🔍 Lecture des meilleurs paramètres HPO depuis s3://{bucket_name}/{params_key}...")
            
            # Utilise les méthodes du SalesPredictionService pour charger les paramètres et les features depuis le même fichier HPO
            hparams = prediction_service._load_best_params(params_key)
            features = prediction_service._load_features(params_key) 
            
            # Nettoyage des clés non-XGBoost et log de l'AUC HPO
            best_auc = hparams.pop("best_auc", None)
            
            # Log des paramètres
            mlflow.log_params(hparams)
            mlflow.log_param("features_used", ", ".join(features))
            if best_auc:
                 mlflow.log_metric("hpo_best_auc", best_auc)
            mlflow.set_tag("data_source_key", data_key)
            
            print(f"✅ Meilleurs paramètres chargés. Caractéristiques: {features}")

            # 3. Lecture et préparation des données d'entraînement depuis S3
            df_raw = s3_csv_to_df(bucket_name=bucket_name, key=data_key)
            
            # Préparation des données (préprocessing)
            print(f"🧹 Préparation des données: {df_raw.shape[0]} lignes initiales.")
            df_final = xgboost_model_logic.preprocess(df_raw) 
            
            # Préparation des jeux d'entraînement finaux (pas de split ici, on utilise tout)
            X_train_final = df_final[features]
            y_train_final = df_final[target_column]
            print(f"✅ Données prêtes pour l'entraînement: {X_train_final.shape[0]} lignes.")
            
            # 4. Entraînement du modèle final
            print("💾 Entraînement du modèle final...")

            final_model = xgboost_model_logic.train_model(
                X_train=X_train_final,
                y_train=y_train_final,
                params=hparams,
                X_test=None, 
                y_test=None,
                custom_threshold=0.5
            )
            
            # 5. Sauvegarde du modèle dans le répertoire SageMaker et Log MLflow
            
            # Sauvegarde locale dans le répertoire de sortie SageMaker (format JSON pour XGBoost)
            Path(model_output_dir).mkdir(parents=True, exist_ok=True)
            model_filename_local = Path(model_output_dir) / "final_xgboost_model.json"
            
            final_model.save_model(str(model_filename_local))
            print(f"✅ Modèle XGBoost enregistré localement dans {model_filename_local}")
            
            # Log de l'artefact dans MLflow
            mlflow_manager.log_model(final_model, name="xgboost_model")

            print("--- Entraînement final XGBoost terminé (MLflow Logged) ---")
            
        except Exception as e:
            error_msg = f"❌ Échec de la fonction train_xgboost_final: {e}"
            print(error_msg)
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final training script for XGBoost using HPO results (S3).")
    
    # NOTE: Ces valeurs sont des chemins d'accès S3
    parser.add_argument("--s3_params_key", type=str, default="params/best_params_xgb.json",
                        help="S3 key pour le fichier JSON des paramètres HPO.")
    parser.add_argument("--s3_data_key", type=str, default="datasets/processed_data.csv",
                        help="S3 key pour le fichier CSV des données d'entraînement.")
    parser.add_argument("--sm_model_dir", type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
                        help="Répertoire de sortie du modèle (convention SageMaker).")
    
    args, _ = parser.parse_known_args()
    
    train_xgboost_final(vars(args))