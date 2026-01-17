import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import optuna
import json
import argparse
import numpy as np  
import os            
from optuna.samplers import TPESampler

from services.optuna_services import OptunaNeuralProphet 
from services.mlflow_manager import MLflowManager      
from typing import Dict, Any, List
import mlflow      
import boto3
import datetime
from pathlib import Path

# --- Variables Globales (Inchangées) ---
s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
s3_prefix = os.getenv('AWS_S3_NEURALP_FOLDER')
s3 = boto3.client('s3')
training_data_filename = f"training_data_20251010.csv"
REGRESSORS = ['active','unit_price'] 
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

def load_data(df) -> pd.DataFrame:
    """ Charge les données d'entraînement réelles et assure le format requis. """    
    try:
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        df['ID'] = df['ID'].astype(str)
    except Exception as e:
        raise ValueError(f"Erreur de conversion de type des colonnes: {e}")
    
    required_cols = ['ds', 'y', 'ID','active','unit_price']
   
    df = df[required_cols]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Le DataFrame doit contenir les colonnes : {required_cols}")
    
    print(f"Données chargées : {len(df)} lignes, {df['ID'].nunique()} séries.")
    return df

# --- Nouvelle fonction pour l'upload S3 (copiée de l'exemple précédent) ---
def upload_file_to_s3(local_path: Path, bucket_name: str, s3_key: str):
    """Télécharge un fichier local vers un emplacement S3 spécifié."""
    print(f"⬆️ Tentative d'upload de {local_path} vers s3://{bucket_name}/{s3_key}")
    try:
        s3_client = boto3.client('s3')
        # S'assure que le chemin est bien une string pour boto3
        s3_client.upload_file(str(local_path), bucket_name, s3_key) 
        print(f"✅ Upload S3 réussi : s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Échec de l'upload S3 : {e}")
        raise
# ------------------------------------------

def run_hpo(n_trials: int, storage_url: str, study_name: str, output_path: str) -> Dict[str, Any]:
    """ Lance l'optimisation HPO de NeuralProphet avec logging MLflow. """
    
    try:
        # 1. SETUP MLFLOW
        mlflow_manager = MLflowManager()
        mlflow_manager.setup() # Configure l'URI et l'Expérience MLflow
        
        # Convertir output_path en Path pour une manipulation facile
        local_output_path = Path(output_path)
        # Déterminer la clé S3 cible à partir du chemin de sortie (par ex. 'params/best_params_neuralp.json')
        s3_target_key = local_output_path.as_posix() # Utilise le chemin relatif comme clé S3

        # --- Démarrage du RUN PARENT MLFLOW pour l'étude Optuna ---
        with mlflow.start_run(run_name="NeuralProphet_HPO_Study"):
            
            # 2. Charger les données et définir les régresseurs
            df = load_csv_from_s3(bucket_name=s3_bucket_name, s3_folder=s3_prefix, file_name=training_data_filename)
            
            if df is None:
                 raise FileNotFoundError(f"Impossible de charger les données d'entraînement depuis S3 : {training_data_filename}")
            
            df = load_data(df)
            
            # Définir les régresseurs que vous souhaitez optimiser
            # NOTE: Ces régresseurs doivent exister dans le df
           
            if not REGRESSORS:
                 print("⚠️ Avertissement: Aucun régresseur (colonne commençant par 'reg') trouvé dans les données.")
            
            # Log des paramètres de l'étude dans le Run Parent
            mlflow.log_param("optuna_n_trials", n_trials)
            mlflow.log_param("optuna_storage", storage_url)
            mlflow.log_param("regressors_used", ", ".join(REGRESSORS) if REGRESSORS else "None")
            mlflow.set_tag("study_name", study_name)
            
            # 3. Initialiser l'objet HPO
            hpo_optimizer = OptunaNeuralProphet(
                df=df, 
                regressors=REGRESSORS, 
                split_days=60 
            )

            # 4. Configurer et lancer l'étude Optuna
            sampler = TPESampler(seed=42)
            
            study = optuna.create_study(
                direction="minimize", 
                study_name=study_name,
                storage=storage_url,
                sampler=sampler,
                load_if_exists=True 
            )
            
            print(f"Lancement de l'optimisation sur {n_trials} essais...")
            
            # Les essais individuels seront logués comme des Nested Runs par hpo_optimizer.objective
            study.optimize(hpo_optimizer.objective, n_trials=n_trials, show_progress_bar=True)

            # 5. Enregistrer les résultats
            best_params = study.best_params
            best_value = study.best_value
            
            print("\n" + "="*50)
            print(f"Optimisation terminée. MAE minimale: {best_value:.4f}")
            
            # Ajouter la meilleure valeur aux paramètres pour l'enregistrement (utilisé par le ModelService)
            result = {
                "best_mae": best_value,
                "parameters": best_params
            }

            # Log de la meilleure métrique dans le Run Parent
            mlflow.log_metric("final_best_val_mae", best_value)
            
            # Enregistrer les meilleurs paramètres dans un fichier JSON (sauvegarde locale)
            with open(local_output_path, 'w') as f:
                json.dump(result, f, indent=4)
                
            print(f"✅ Résultats enregistrés localement dans : {local_output_path}")

            # --- AJOUT: Upload vers S3 ---
            if s3_bucket_name:
                upload_file_to_s3(
                    local_path=local_output_path, 
                    bucket_name=s3_bucket_name, 
                    s3_key=s3_target_key # e.g., 'params/best_params_neuralp.json'
                )
            else:
                print("❌ AWS_S3_BUCKET_NAME n'est pas défini. L'upload S3 des paramètres est ignoré.")
            # ---------------------------

            print("="*50)
            return result
        
    except Exception as e:
        print(f"❌ Échec de la fonction run_hpo pour NeuralProphet: {e}")
        # Log de l'échec dans MLflow si possible
        try:
            mlflow.set_tag("status", "failed")
        except:
            pass # Si le run n'a pas pu être démarré
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for NeuralProphet using Optuna.")
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50, 
        help='Number of trials to run in Optuna HPO.'
    )
    parser.add_argument(
        '--storage_url', 
        type=str, 
        default='sqlite:///neuralp_hpo.db', 
        help='Optuna storage URL (e.g., sqlite:///db.db or postgresql://user:pass@host/db).'
    )
    parser.add_argument(
        '--study_name', 
        type=str, 
        default='neuralp_multi_ts_study', 
        help='Name of the Optuna study.'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='params/best_params_neuralp.json', 
        help='Path to save the best hyperparameters found.'
    )
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    run_hpo(
        n_trials=args.n_trials,
        storage_url=args.storage_url,
        study_name=args.study_name,
        output_path=args.output_path
    )