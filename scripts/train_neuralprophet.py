import pandas as pd
import optuna
import json
import argparse
import numpy as np  
import os            
from optuna.samplers import TPESampler

from services.optuna_services import OptunaNeuralProphet 
from services.mlflow_manager import MLflowManager      
from typing import Dict, Any, List, Optional
import mlflow                           
from pathlib import Path 
           

# --- Constantes et valeurs par défaut ---
DEFAULT_REGRESSORS = ["reg1", "reg2", "reg3", "reg4"] # Exemple de régresseurs
DEFAULT_STORAGE_URL = 'None' 
DEFAULT_STUDY_NAME = 'neuralp_multi_ts_study'
DEFAULT_OUTPUT_PATH = 'results/best_params_neuralp.json'
DEFAULT_DATA_PATH = "data/processed_multi_ts.csv"


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """ Charge les données d'entraînement réelles et assure le format requis. """
    print(f"Chargement des données réelles depuis: {path}")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERREUR: Fichier non trouvé à {path}.")
        raise
        
    # Conversion des types (Crucial pour NeuralProphet)
    try:
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        df['ID'] = df['ID'].astype(str)
    except Exception as e:
        raise ValueError(f"Erreur de conversion de type des colonnes: {e}")
    
    required_cols = ['ds', 'y', 'ID']
    if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Le DataFrame doit contenir les colonnes : {required_cols}")
    
    print(f"Données chargées : {len(df)} lignes, {df['ID'].nunique()} IDs uniques.")
    return df


def run_hpo(args: Dict[str, Any]):
    
    print("="*50)
    print("--- Démarrage de la recherche HPO Optuna pour NeuralProphet ---")

    try:
        mlflow_manager = MLflowManager()
        mlflow_manager.setup()

        with mlflow.start_run(run_name="NeuralProphet_HPO_Study"):
            
            # 1. Parsing des arguments
            n_trials = int(args.get("n_trials", 50))
            data_path = args.get("data_path", DEFAULT_DATA_PATH)
            output_path = args.get("output_path", DEFAULT_OUTPUT_PATH)
            
            # Gestion de l'URL de stockage : convertit la chaîne 'None' en Python None
            storage_url_arg = args.get("storage_url", DEFAULT_STORAGE_URL)
            storage_url: Optional[str] = storage_url_arg if storage_url_arg and storage_url_arg.lower() != 'none' else None
            study_name = args.get("study_name", DEFAULT_STUDY_NAME)
            
            # Récupération de la liste des régresseurs externes
            regressors = json.loads(args.get("regressors", json.dumps(DEFAULT_REGRESSORS)))

            if not regressors:
                print("⚠️ Avertissement: Aucun régresseur externe spécifié.")
                
            mlflow.log_param("optuna_n_trials", n_trials)
            mlflow.log_param("regressors_used", ", ".join(regressors))
            mlflow.log_param("optuna_storage_url", storage_url if storage_url else "in-memory (no DB)")
            mlflow.log_param("optuna_study_name", study_name)
            mlflow.set_tag("model_type", "NeuralProphet")
            
            # 2. Chargement des données
            df = load_data(data_path)

            # 3. Instanciation et exécution du service HPO
            optuna_np = OptunaNeuralProphet(
                df=df, 
                regressors=regressors,
            )

            print(f"🧠 Lancement de la recherche HPO Optuna (n_trials={n_trials}) en mode {'in-memory' if storage_url is None else 'stocké'}...")

            # Appel direct à l'optimiseur avec les paramètres Optuna
            best_params , best_value = optuna_np.optimize(
                n_trials=n_trials,
                storage_url=storage_url, # Passera None si 'None' a été parsé
                study_name=study_name
            )

            result = {
                "best_mae": best_value,
                "parameters": best_params,
                "regressors": regressors
            }

            print(f"✅ HPO complété. Meilleur score (MAE): {best_value:.4f}")
            print("💾 Sauvegarde des meilleurs paramètres...")

            mlflow.log_metric("final_best_val_mae", best_value)
           
            # Sauvegarde des résultats
            full_output_path = Path(output_path)
            full_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_output_path, 'w') as f:
                json.dump(result, f, indent=4)

            print(f"✅ Meilleurs paramètres sauvegardés dans : {full_output_path}")
            print("="*50)
            return result
        
    except Exception as e:
        print(f"❌ Échec de la fonction run_hpo pour NeuralProphet: {e}")
        try:
            mlflow.set_tag("status", "failed")
        except:
            pass 
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for NeuralProphet using Optuna.")
    
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50, 
        help='Number of trials to run in Optuna HPO.'
    )
    # Modification: Valeur par défaut 'None' pour le mode in-memory
    parser.add_argument(
        '--storage_url', 
        type=str, 
        default=DEFAULT_STORAGE_URL, 
        help='Optuna storage URL (e.g., sqlite:///db.db or None for in-memory).'
    )
    parser.add_argument(
        '--study_name', 
        type=str, 
        default=DEFAULT_STUDY_NAME, 
        help='Name of the Optuna study.'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=DEFAULT_OUTPUT_PATH, 
        help='Path to save the best hyperparameters found.'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default=DEFAULT_DATA_PATH, 
        help='Path to the processed data CSV file.'
    )
    # Ajout de l'argument pour les régresseurs
    parser.add_argument(
        '--regressors', 
        type=str, 
        default=json.dumps(DEFAULT_REGRESSORS), 
        help='JSON list of external regressors columns to include (e.g., ["reg1", "reg2"]).'
    )
    
    args = parser.parse_args()
    
    run_hpo(vars(args))