import mlflow
import os
import xgboost as xgb
from neuralprophet import NeuralProphet as NP
import warnings
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

class MLflowManager:
    
    def __init__(
        self
    ):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI") 
        self.experiment_name = "Djoli ML"
        MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
        MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        self.client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

    def setup(self):
        """ Configure l'URI et l'Expérience MLflow. """
        print(f"Setting MLflow tracking URI to: {self.tracking_uri}")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            print(f"Creating new MLflow experiment: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)
        else:
            print(f"Using existing MLflow experiment: {self.experiment_name}")
            mlflow.set_experiment(self.experiment_name)

    def log_run_params(self, params: dict):
        mlflow.log_params(params)
    
    def log_run_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics) 
    
    def log_model(self, model, name: str, **kwargs):
       
        if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.Booster):
            mlflow.xgboost.log_model(xgb_model=model, artifact_path=name, **kwargs)
        
        elif isinstance(model, NP):
           
            model_path = "neuralprophet_model.pt"
            model.save(model_path)
            
            # 2. Log de l'artefact
            mlflow.log_artifact(model_path, artifact_path=name)
            
            # 3. Nettoyage
            os.remove(model_path)
            print(f"NeuralProphet model logged as artifact at: {name}")
        else:
            raise TypeError(f"Unsupported model type: {type(model)}.")

    def get_latest_model_path(self, model_name: str) -> Optional[str]:
        # Logique pour récupérer le chemin (inchangée)
        try:
            latest_version = self.client.get_latest_versions(name=model_name, stages=["Production", "Staging", "None"])[0]
            return latest_version.source
        except IndexError:
            print(f"No model found with name '{model_name}'.")
            return None