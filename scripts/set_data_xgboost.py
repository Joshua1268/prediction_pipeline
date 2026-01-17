import sys
import os

# Ajouter le dossier parent du script au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.model_services import SalesPredictionService
from services.models.xgboost import XgboostPredictor
from dotenv import load_dotenv
import os
import datetime
import boto3

load_dotenv()

s3_bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
s3_prefix = os.getenv('AWS_S3_XGBOOST_FOLDER')
s3 = boto3.client('s3')

sales_service = SalesPredictionService()
xgboost_service = XgboostPredictor(s3_bucket_name, s3_prefix)


today = datetime.date.today().strftime("%Y%m%d")



def set_data(bucket_name: str, prefix: str):
    print("⏳ Préparation et upload des données...")
    df = sales_service.load_data()
    print (df)
    print(df.columns)
    df = xgboost_service.preprocess(df)
    xgboost_service.prepare_and_upload_data(df)
    print(f"✅ Nouvelles données uploadées vers s3://{bucket_name}/{prefix}")


def clean_s3_data(bucket_name: str, prefix: str):
    """Vide tous les objets dans le préfixe S3 spécifié."""
    s3 = boto3.client('s3')

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' not in response:
        print(f"🧹 Aucun fichier à nettoyer dans s3://{bucket_name}/{prefix}")
        return

    to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
    
    # Sécurité: S'assurer qu'on a bien des clés à supprimer avant l'appel
    if to_delete:
        s3.delete_objects(
            Bucket=bucket_name,
            Delete={'Objects': to_delete}
        )
        print(f"✅ Dossier vidé : s3://{bucket_name}/{prefix}")
    else:
        print(f"🧹 Aucun fichier à nettoyer dans s3://{bucket_name}/{prefix}")
    
# --- Logique Principale d'Exécution ---

if __name__ == "__main__":
    
    # 1. Initialiser le drapeau de vérification
    file_found = False
    
    print(f"Recherche du fichier pour la date : {today}.csv")
    
    # 2. Lister les objets dans le bucket S3
    response = s3.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_prefix)
    
    # 3. Parcourir et vérifier l'existence du fichier du jour
    for obj in response.get('Contents', []):
        file_name = obj['Key']
        
        # Vérifie si le nom du fichier se termine par la date du jour
        if file_name.endswith(f"{today}.csv"):
            print(f"✅ Le fichier du jour trouvé : {file_name}")
            file_found = True # Fichier trouvé !
            break             # Inutile de continuer la boucle

    # 4. Action Conditionnelle (Nettoyage et Upload)
    if not file_found:
        print(f"❌ Le fichier du jour n'a pas été trouvé ou le dossier est vide. Déclenchement de la mise à jour...")
        
        # Nettoyage de l'ancien contenu (s'il existe)
        clean_s3_data(s3_bucket_name, s3_prefix) 
        
        # Upload des nouvelles données
        set_data(s3_bucket_name, s3_prefix) 

    else:
        print("Opération terminée : le fichier du jour était déjà en place. Aucune action requise.")