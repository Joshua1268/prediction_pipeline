import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
import numpy as np
import warnings
import xgboost as xgb
from neuralprophet import NeuralProphet as NP, set_random_seed
from typing import Dict, Any, Tuple, Optional, Union, List
import mlflow 
import optuna

warnings.filterwarnings("ignore")

# Configuration de la seed pour la reproductibilité
np.random.seed(42)
set_random_seed(0)


class OptunaNeuralProphet:
    
    def __init__(self, df: pd.DataFrame, regressors: list, split_days: int = 60):
        self.df = df
        self.regressors = regressors
        self.split_days = split_days
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        
        if 'ID' not in df.columns:
            raise ValueError("Le DataFrame doit contenir une colonne 'ID' pour la modélisation multi-séries (Global/Local).")


    def _prepare_data_split(self) -> None:
        """ Effectue le prétraitement et le split des données (Train/Validation). """
        df_temp = self.df.copy()
        splitter_model = NP() 
        
        try:
            time_span = (df_temp["ds"].max() - df_temp["ds"].min()).days
            split_ratio = self.split_days / time_span if time_span > 0 else 0.2
        except Exception as e:
             print(f"⚠️ Avertissement: Erreur de calcul de l'étendue temporelle: {e}. Utilisation de split_ratio=0.2.")
             split_ratio = 0.2
        
        try:
            self.df_train, self.df_test = splitter_model.split_df(
                df_temp, 
                valid_p=split_ratio, 
                local_split=True 
            )

            if self.df_train.empty or self.df_test.empty:
                 raise ValueError("Split de données NeuralProphet a résulté en un ensemble d'entraînement ou de test vide.")

            print(f"Data split for HPO (Multi-ID): Train={len(self.df_train)}, Test={len(self.df_test)}")
        except Exception as e:
             print(f"❌ Erreur lors du split des données NeuralProphet: {e}")
             raise

    def objective(self, trial) -> float:
        """ Fonction objectif Optuna pour NeuralProphet avec logging MLflow. """
        
        # --- Démarrage du Nested Run MLflow ---
        with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True): 
            try:
                if self.df_train is None:
                    self._prepare_data_split()

                n_lags_max = min(24, len(self.df_train) // 2) if hasattr(self, 'df_train') else 7
                n_lags_max = max(7, n_lags_max)

                # --- Définition des paramètres HPO ---
                trend_mode = trial.suggest_categorical("trend_global_local", ["global", "local"])
                season_mode = trial.suggest_categorical("season_global_local", ["global", "local"])

                params = {
                    "n_changepoints": trial.suggest_int("n_changepoints", 5, 100),
                    'changepoints_range': trial.suggest_float('changepoints_range', 0.8, 0.95),
                    "trend_reg": trial.suggest_float("trend_reg", 0, 5),
                    "trend_global_local": trend_mode, 
                    "yearly_seasonality": trial.suggest_categorical("yearly_seasonality", [True, False, 6]),
                    "weekly_seasonality": trial.suggest_categorical("weekly_seasonality", [True, False, 3]),
                    "daily_seasonality": trial.suggest_categorical("daily_seasonality", [True, False, 6]),
                    "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
                    "seasonality_reg": trial.suggest_float("seasonality_reg", 0.1, 100.0, log=True),
                    "season_global_local": season_mode,
                    'n_lags': trial.suggest_int('n_lags', 8, n_lags_max),
                    'n_forecasts': trial.suggest_int('n_forecasts', 7, 30),
                    "ar_layers": trial.suggest_categorical("ar_layers", [None, [16], [32], [16, 8]]),
                    "lagged_reg_layers": trial.suggest_categorical("lagged_reg_layers", [[16], [32], [16, 8]]),
                    "trend_local_reg": trial.suggest_float("trend_local_reg", 0.0, 1.0) if trend_mode == "local" else None,
                    "seasonality_local_reg": trial.suggest_float("seasonality_local_reg", 0.0, 1.0) if season_mode == "local" else None,
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                    "epochs": trial.suggest_int("epochs", 50, 200),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                    "loss_func": trial.suggest_categorical("loss_func", ["MSE", "SmoothL1Loss"])
                }

                params = {k: v for k, v in params.items() if v is not None}
                add_holidays = trial.suggest_categorical("add_ci_holidays", [True, False])
                
                # Log des hyperparamètres HPO
                mlflow.log_params(params)
                mlflow.log_param("add_ci_holidays", add_holidays)
                mlflow.set_tag("model_type", "NeuralProphet")


                model = NP(**params)

                if add_holidays:
                    model = model.add_country_holidays("CI")

                for reg in self.regressors:
                    n_lags_key = f"{reg}_n_lags" 
                    n_lags_reg = trial.suggest_int(n_lags_key, 2, 20)
                    model = model.add_lagged_regressor(reg, n_lags=n_lags_reg)
                    mlflow.log_param(n_lags_key, n_lags_reg) # Log des lags pour les régresseurs

                metrics = model.fit(self.df_train, validation_df=self.df_test, freq="D", progress="none", early_stopping=True)
                
                # Vérifiez si le fit a réussi avant de prédire
                if metrics is None or metrics.empty:
                    raise RuntimeError("L'entraînement du modèle NeuralProphet a échoué (metrics vides).")

                forecast = model.predict(self.df_test)

                merged_df = pd.merge(self.df_test[['ds', 'y', 'ID']], forecast[['ds', 'yhat1', 'ID']], on=['ds', 'ID'], how='inner')
                merged_df.dropna(subset=['y', 'yhat1'], inplace=True)

                if merged_df.empty:
                    raise ValueError("Aucune donnée valide après la prédiction et la fusion pour le calcul du MAE.")

                mae = mean_absolute_error(merged_df['y'], merged_df['yhat1'])
                
                mlflow.log_metric("val_mae", mae)
                
                return mae
                
            except Exception as e:
                # Log MAE = inf si l'essai échoue
                error_msg = f"❌ Échec de l'essai Optuna {trial.number}: {e}"
                print(error_msg)
                mlflow.log_metric("val_mae", float('inf')) 
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:250])
                return float('inf')

        
    # CORRECTION: Mise à jour de la signature et utilisation de storage/study_name
    def optimize(self, n_trials: int, storage_url: str, study_name: str) -> Tuple[Dict[str, Any], float]:
        """ Lance l'optimisation Optuna et retourne les meilleurs hyperparamètres et score (MAE). """
        
        try:
            if self.df_train is None:
                self._prepare_data_split()
                
            # Utilisation de storage_url et study_name
            study = optuna.create_study(
                direction="minimize",
                storage=storage_url,
                study_name=study_name
            ) 
            print(f"Démarrage de l'optimisation Optuna pour NeuralProphet avec {n_trials} essais...")
            
            study.optimize(self.objective, n_trials=n_trials) # Utilisation de n_trials
            
            print("\nOptimisation terminée.")
            print(f"Meilleur MAE: {study.best_value}")
            print("Meilleurs hyperparamètres:")
            print(study.best_params)
            
            # Retourne les paramètres et la meilleure valeur (MAE)
            return study.best_params, study.best_value
        except Exception as e:
            print(f"❌ Échec de l'étude Optuna pour NeuralProphet: {e}")
            raise


class OptunaXgboost:
   
    def __init__(self, df: pd.DataFrame, features: List[str], test_days: int = 60):
        self.df = df
        self.features = features
        self.test_days = test_days
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None


    def _prepare_data_split(self):
        """ Effectue le split des données (Train/Validation) basé sur la date. """
        try:
            df_temp = self.df.copy()

            if not pd.api.types.is_datetime64_any_dtype(df_temp["delivery_date"]):
                 df_temp["delivery_date"] = pd.to_datetime(df_temp["delivery_date"])

            last_date = df_temp["delivery_date"].max()
            split_date = last_date - pd.Timedelta(days=self.test_days)

            train = df_temp[df_temp["delivery_date"] < split_date].copy()
            test = df_temp[df_temp["delivery_date"] >= split_date].copy().head(len(train) // 4) # Limiter la taille du jeu de test

            if train.empty or test.empty:
                raise ValueError(f"Erreur de split: l'ensemble d'entraînement ({len(train)}) ou de validation ({len(test)}) est vide. Vérifiez la colonne 'delivery_date'.")

            # Assurez-vous que toutes les features sont présentes
            missing_cols = [f for f in self.features if f not in train.columns or f not in test.columns]
            if missing_cols:
                 raise ValueError(f"Colonnes de features manquantes dans les données splitées: {missing_cols}")

            self.X_train = train[self.features]
            self.y_train = train['is_ordered']
            self.X_val = test[self.features]
            self.y_val = test['is_ordered']

            print(f"Data split for HPO: Train={len(self.X_train)}, Val={len(self.X_val)}")
        except Exception as e:
             print(f"❌ Erreur lors du split des données XGBoost: {e}")
             raise
        

    def objective(self, trial) -> float:
        """ Fonction objectif Optuna pour XGBoost avec logging MLflow. """
        
        
        with mlflow.start_run(run_name=f"Trial-{trial.number}", nested=True): 
            try:
                if self.X_train is None:
                    self._prepare_data_split()
                
                # --- Définition des paramètres HPO ---
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "random_state": 42
                }
                
                mlflow.log_params(params)
                mlflow.set_tag("model_type", "XGBoost")
                
                model = xgb.XGBClassifier(
                    **params,
                    early_stopping_rounds=10  
                )
                    
                model.fit(
                    self.X_train,
                    self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    verbose=False,
                )
                
                if model.best_iteration != -1:
                    mlflow.log_metric("best_iteration", model.best_iteration)

                preds_proba = model.predict_proba(self.X_val)[:, 1]
                auc_score = roc_auc_score(self.y_val, preds_proba)


                mlflow.log_metric("val_auc", auc_score)
                
                return auc_score
            
            except Exception as e:
                error_msg = f"❌ Échec de l'essai Optuna {trial.number}: {e}"
                print(error_msg)
                mlflow.log_metric("val_auc", 0.0) # AUC par défaut pour échec
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(e)[:250])
                return 0.0

            
    # CORRECTION: Mise à jour de la signature et utilisation de storage/study_name
    def optimize(self, n_trials: int, storage_url: str, study_name: str) -> Tuple[Dict[str, Any], float]:
        """ Lance l'optimisation Optuna et retourne les meilleurs hyperparamètres et score (AUC). """
        
        try:
            if self.X_train is None:
                 self._prepare_data_split()
            
            # Utilisation de storage_url et study_name
            study = optuna.create_study(
                direction="maximize",
                storage=storage_url,
                study_name=study_name
            ) 
            print(f"Démarrage de l'optimisation Optuna pour XGBoost avec {n_trials} essais...")
            
            study.optimize(self.objective, n_trials=n_trials) # Utilisation de n_trials

            print("\nOptimisation terminée.")
            print(f"Meilleure AUC: {study.best_value}")
            print("Meilleurs hyperparamètres:")
            print(study.best_params)
            
            # Retourne les paramètres et la meilleure valeur (AUC)
            return study.best_params , study.best_value
        except Exception as e:
            print(f"❌ Échec de l'étude Optuna pour XGBoost: {e}")
            raise