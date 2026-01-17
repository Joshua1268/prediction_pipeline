import sagemaker
import os
import subprocess
from datetime import datetime
from pathlib import Path
import dotenv
import boto3

dotenv.load_dotenv()

# --- Configurations ---
aws_region = os.getenv("AWS_REGION", os.getenv("AWS_REGION_NAME"))
if not aws_region:
    raise ValueError("La région AWS ('AWS_REGION' ou 'AWS_REGION_NAME') doit être définie.")

boto_session = boto3.Session(region_name=aws_region)
session = sagemaker.Session(boto_session=boto_session)
account_id = boto_session.client("sts").get_caller_identity()["Account"]

# Nom du repository ECR pour votre code
ECR_REPOSITORY_NAME = "sagemaker-mlops-hpo-image" 
# ✅ NOUVEAU TAG : Spécifier l'architecture pour le tag
IMAGE_TAG = "amd64-latest" 

# URI ECR complet qui sera utilisé par les jobs SageMaker suivants
ECR_URI = f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com/{ECR_REPOSITORY_NAME}:{IMAGE_TAG}"

CURRENT_FILE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = CURRENT_FILE_DIR.resolve()

# --- Fonctions Docker/ECR ---
def build_and_push_image(ecr_uri: str, source_dir: Path, region: str):
    """Construit l'image Docker localement pour linux/amd64 et la pousse vers AWS ECR."""
    
    # 1. Authentification Docker auprès d'ECR
    print(f"1. Authentification Docker auprès d'ECR dans la région {region}...")
    
    auth_cmd = f"aws ecr get-login-password --region {region}"
    try:
        login_password = subprocess.run(
            auth_cmd, shell=True, check=True, capture_output=True, text=True
        ).stdout.strip()
        
        login_cmd = f"docker login --username AWS --password {login_password} {account_id}.dkr.ecr.{region}.amazonaws.com"
        # Ajout de stdout=subprocess.DEVNULL au login pour ne pas afficher le mot de passe dans les logs
        subprocess.run(login_cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        
        print("   -> Authentification réussie.")
    except Exception as e:
        print(f"ERREUR d'authentification ECR: Assurez-vous d'avoir AWS CLI et Docker installés et configurés. {e}")
        return

    # 2. Construction et Push de l'image Docker (Combiné via buildx)
    # Nous utilisons buildx pour cibler explicitement linux/amd64
    print(f"\n2. Construction et Push de l'image Docker pour linux/amd64: {ecr_uri}")
    
    # Utilisation de docker buildx pour la construction multi-plateforme et le push direct
    # Note: L'utilisateur doit avoir 'docker buildx install' exécuté au moins une fois
    build_push_cmd = (
        f"docker buildx build "
        f"--platform linux/amd64 "
        f"--tag {ecr_uri} "
        f"--push "
        f"{source_dir}"
    )

    try:
        subprocess.run(build_push_cmd, shell=True, check=True)
        print("   -> Construction et Push vers ECR réussis (architecture AMD64). L'image est maintenant disponible pour SageMaker.")
    except Exception as e:
        print(f"ERREUR lors du build/push AMD64. Assurez-vous que docker buildx est installé et que vous avez les permissions : {e}")
        return


# --- Main ---
if __name__ == "__main__":
    
    print(f"🚀 Déploiement de l'environnement MLOps (Code et Dépendances) vers ECR.")
    print(f"Le code sera poussé vers : {ECR_URI}")
    
    # Crée le repository ECR s'il n'existe pas (étape cruciale)
    ecr_client = boto_session.client("ecr", region_name=aws_region)
    try:
        ecr_client.create_repository(repositoryName=ECR_REPOSITORY_NAME)
        print(f"\nRepository ECR '{ECR_REPOSITORY_NAME}' créé (ou existant).")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"\nRepository ECR '{ECR_REPOSITORY_NAME}' existe déjà. Poursuite...")
    except Exception as e:
        print(f"Erreur lors de la vérification/création du repository ECR : {e}")
        exit()
        
    # Lance le processus de construction et de push
    build_and_push_image(
        ecr_uri=ECR_URI, 
        source_dir=SOURCE_DIR,
        region=aws_region
    )

    print("\n✅ Le déploiement est terminé.")
    print(f"UTILISEZ CET URI dans votre service Django-Q pour les jobs HPO/Training : \n{ECR_URI}")