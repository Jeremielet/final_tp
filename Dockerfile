# Dockerfile pour l'API de détection de fraude

FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p artifacts/models data/raw mlruns

# Exposer le port 8002
EXPOSE 8002

# Commande pour lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002"]
