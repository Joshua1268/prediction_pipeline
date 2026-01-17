import pandas as pd
from prediction_model import ProductPredictionModel
from model_trainer import ModelTrainer

def test_prediction_pipeline():
    """Test simple pour vérifier que le pipeline de prédiction fonctionne"""
    print("Test du pipeline de prédiction...")
    
    # Initialiser le trainer et charger/entraîner le modèle
    trainer = ModelTrainer()
    predictor = trainer.load_or_train_model()
    
    # Tester une prédiction
    result = predictor.get_prediction_for_product("Tomate", periods=7)
    
    print(f"Prédictions pour {result['product']}:")
    for record in result['predictions'][:3]:  # Afficher les 3 premiers jours
        date = record['ds']
        pred = record['yhat1']
        print(f"  {date}: {pred:.2f}")
    
    # Obtenir des insights
    insights = predictor.get_insights()
    print(f"\nTendance: {insights['trend']}")
    print(f"Moyenne actuelle: {insights['latest_average_actual']:.2f}")
    print(f"Moyenne prévue: {insights['latest_average_forecast']:.2f}")
    
    print("\nTest réussi!")

if __name__ == "__main__":
    test_prediction_pipeline()