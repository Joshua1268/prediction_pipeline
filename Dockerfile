# =======================================================
# STAGE 1 : BUILDER (Contient tous les outils lourds)
# =======================================================
FROM python:3.11-slim-bookworm AS builder

# 1. Installer les outils de construction système (pour les dépendances Python)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Définir le répertoire de travail et copier les dépendances
WORKDIR /app
COPY requirements.txt .

# 3. Installer toutes les dépendances
# Mise à jour de pip pour un meilleur résolveur de dépendances
RUN pip install --upgrade pip
# Utilise --no-cache-dir pour éviter de stocker les fichiers temporaires pip
RUN pip install --no-cache-dir -r requirements.txt

# =======================================================
# STAGE 2 : FINAL (Image Minimale)
# =======================================================
# On repart de l'image de base légère, mais SANS les outils de compilation
FROM python:3.11-slim-bookworm

# 1. Copier uniquement les paquets Python installés
# Cette méthode de copie est plus fiable pour les images slim/aarch64
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 2. Configurer l'environnement SageMaker
WORKDIR /opt/ml/code

# Copie le code de l'application DANS l'image ECR
COPY . /opt/ml/code

# 3. Configurer le PYTHONPATH
# Correction : Définit la variable sans référence à l'ancien $PYTHONPATH pour éviter l'avertissement.
ENV PYTHONPATH=/opt/ml/code

# 4. Spécification finale (recommandée mais optionnelle pour les Scripts/Processing Jobs)
# CMD ["python3", "./votre_script_principal.py"] 
# REMARQUE : 'command' dans ScriptProcessor.run() outrepasse cette CMD.