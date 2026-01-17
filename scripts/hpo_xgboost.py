import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import pandas as pd
from pathlib import Path
import mlflow


from repositories.sales_data import SalesRepository
from transformations.sales_data_builder import SalesDataBuilder 
from services.models.xgboost import XgboostPredictor 
from services.optuna_services import OptunaXgboost
from services.mlflow_manager import MLflowManager
from typing import Dict, Any, List, Optional
import os
import boto3
import datetime

# --- Variables Globales (Inchangées) ---
S3_PARAMS_PREFIX = "params" # Le préfixe S3 souhaité pour les paramètres
DEFAULT_OUTPUT_FILENAME = f"{S3_PARAMS_PREFIX}/best_params_xgb.json" # Changé pour inclure le préfixe
# La clé S3 complète sera construite à partir de DEFAULT_OUTPUT_FILENAME
S3_TARGET_KEY = f"{S3_PARAMS_PREFIX}/best_params_xgb.json" # Clé cible S3

DEFAULT_STORAGE_URL = 'None' 
DEFAULT_STUDY_NAME = 'xgboost_study'
DEFAULT_FEATURES = ["day_of_week", "week", "standard_id", "month_of_year", "active_lag1", "stockout_lag7", "is_holiday"]
s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
s3_prefix = os.getenv('AWS_S3_XGBOOST_FOLDER')
s3 = boto3.client('s3')
training_data_filename = f"training_data_20251010.csv"
# ----------------------------------------

def load_csv_from_s3(bucket_name: str, s3_folder: str, file_name: str, local_dir: str = "./tmp"):
    
    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)

    s3_key = os.path.join(s3_folder, file_name).replace("\\", "/")
    local_path = os.path.join(local_dir, file_name)

    try:
        # Téléchargement du fichier
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"✅ Fichier téléchargé : {s3_key}")

        # Lecture du CSV
        df = pd.read_csv(local_path)
        print(f"📄 CSV chargé en DataFrame ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        return df

    except Exception as e:
        print(f"❌ Erreur lors du téléchargement ou de la lecture : {e}")
        return None
# --- Nouvelle fonction pour l'upload S3 ---
def upload_file_to_s3(local_path: Path, bucket_name: str, s3_key: str):
    """Télécharge un fichier local vers un emplacement S3 spécifié."""
    print(f"⬆️ Tentative d'upload de {local_path} vers s3://{bucket_name}/{s3_key}")
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_path), bucket_name, s3_key)
        print(f"✅ Upload S3 réussi : s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Échec de l'upload S3 : {e}")
        raise
# ------------------------------------------

def run_hpo_xgboost(args: Dict[str, Any]):
    
    try:
        print("--- Démarrage de la recherche HPO Optuna pour XGBoost (Autonome) ---")

        mlflow_manager = MLflowManager()
        mlflow_manager.setup()

        with mlflow.start_run(run_name="XGBoost_HPO_Study"):

            features = json.loads(args.get("features", json.dumps(DEFAULT_FEATURES)))
            n_trials = int(args.get("n_trials", 50))
            
            storage_url_arg = args.get("storage_url", DEFAULT_STORAGE_URL)
            storage_url: Optional[str] = storage_url_arg if storage_url_arg and storage_url_arg.lower() != 'none' else None
            
            study_name = args.get("study_name", DEFAULT_STUDY_NAME)
            
            sagemaker_output_dir = args.get("sm_output_data_dir", "params")
            output_filename = args.get("output_filename", DEFAULT_OUTPUT_FILENAME) # 'params/best_params_xgb.json'

            if not features:
                raise ValueError("La liste des 'features' n'a pas été fournie.")

            mlflow.log_param("optuna_n_trials", n_trials)
            mlflow.log_param("features_used", ", ".join(features))
            mlflow.log_param("optuna_storage_url", storage_url if storage_url else "in-memory (no DB)")
            mlflow.log_param("optuna_study_name", study_name)
            mlflow.set_tag("model_type", "XGBoost")

            try:
                print("🚀 Démarrage du pipeline de données: DB -> Preprocessing pour XGBoost...")
                
                # Le reste du pipeline de données est implicitement géré par load_csv_from_s3
                
            except Exception as e:
                print(f"❌ Échec du pipeline de données (DB/Preprocessing): {e}")
                raise

            # 3. Instanciation et exécution du service HPO
            # Assurez-vous que s3_bucket_name et s3_prefix sont définis
            if not s3_bucket_name or not s3_prefix:
                print("⚠️ AWS_S3_BUCKET_NAME ou AWS_S3_XGBOOST_FOLDER n'est pas défini dans l'environnement.")
                # Laissez-le lever une erreur plus tard si le chargement échoue, mais avertissez.

            df_training = load_csv_from_s3(s3_bucket_name, s3_prefix, training_data_filename)

            if df_training is None:
                 raise FileNotFoundError(f"Impossible de charger les données d'entraînement depuis S3 : {training_data_filename}")
                 
            optuna_xgb = OptunaXgboost(
                df=df_training, 
                features=features,
            )

            print(f"🧠 Lancement de la recherche HPO Optuna (n_trials={n_trials}) en mode {'in-memory' if storage_url is None else 'stocké'}...")

            # Appel à l'optimiseur avec l'URL de stockage ajustée (None ou URI)
            best_params , best_value = optuna_xgb.optimize(
                n_trials=n_trials,
                storage_url=storage_url,
                study_name=study_name
            )

            result = {
                "best_auc": best_value,
                "parameters": best_params,
                "features": features
            }

            print(f"✅ HPO complété. Meilleur score (AUC): {best_value:.4f}")
            print("💾 Sauvegarde des meilleurs paramètres...")

            mlflow.log_metric("final_best_val_auc", best_value)
           
            # Sauvegarde des résultats
            # full_output_path est le chemin local (e.g., params/params/best_params_xgb.json)
            full_output_path = Path(sagemaker_output_dir) / output_filename

            full_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_output_path, 'w') as f:
                json.dump(result, f, indent=4)

            print(f"✅ Meilleurs paramètres sauvegardés localement dans {full_output_path}")

            # --- AJOUT: Upload vers S3 ---
            if s3_bucket_name:
                # La clé cible S3 doit être uniquement 'params/best_params_xgb.json'
                # On utilise la variable globale S3_TARGET_KEY définie comme "params/best_params_xgb.json"
                upload_file_to_s3(
                    local_path=full_output_path, 
                    bucket_name=s3_bucket_name, 
                    s3_key=S3_TARGET_KEY # 'params/best_params_xgb.json'
                )
            else:
                print("❌ AWS_S3_BUCKET_NAME n'est pas défini. L'upload S3 des paramètres est ignoré.")
            # ---------------------------
            
        print("--- HPO XGBoost terminé ---")
        
    except Exception as e:
        print(f"❌ Échec de la fonction run_hpo_xgboost: {e}")
        try:
            mlflow.set_tag("status", "failed")
        except:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for XGBoost using Optuna.")

    # Arguments HPO
    parser.add_argument("--features", type=str, default=json.dumps(DEFAULT_FEATURES))
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--output_filename", type=str, default=DEFAULT_OUTPUT_FILENAME)
    
    # ARGUMENTS Optuna MIS À JOUR
    parser.add_argument(
        '--storage_url', 
        type=str, 
        default='None', # Nouvelle valeur par défaut pour in-memory
        help='Optuna storage URL (e.g., sqlite:///db.db or None for in-memory).'
    )
    parser.add_argument(
        '--study_name', 
        type=str, 
        default=DEFAULT_STUDY_NAME, 
        help='Name of the Optuna study.'
    )

    # Argument SageMaker
    parser.add_argument("--sm-output-data-dir", type=str, default="params", dest="sm_output_data_dir")

    args, _ = parser.parse_known_args()

    run_hpo_xgboost(vars(args))