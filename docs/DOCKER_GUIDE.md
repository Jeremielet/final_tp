# Guide Docker - Fraud Detection API

Ce guide explique comment utiliser Docker pour déployer l'API de détection de fraude.

## 🐳 Prérequis

- Docker installé (version 20.10+)
- Docker Compose installé (version 2.0+)

## 🚀 Démarrage rapide

### Option 1 : Docker Compose (Recommandé)

```bash
# Construire et démarrer l'API
docker-compose up -d

# Vérifier que le conteneur est lancé
docker-compose ps

# Voir les logs
docker-compose logs -f

# Arrêter l'API
docker-compose down
```

L'API sera disponible sur http://localhost:8002

### Option 2 : Docker seul

```bash
# Construire l'image
docker build -t fraud-detection-api .

# Lancer le conteneur
docker run -d \
  --name fraud-detection-api \
  -p 8002:8002 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  fraud-detection-api

# Voir les logs
docker logs -f fraud-detection-api

# Arrêter le conteneur
docker stop fraud-detection-api
docker rm fraud-detection-api
```

## 📦 Structure Docker

### Dockerfile

Le Dockerfile utilise :
- **Base image**: `python:3.12-slim` (légère et optimisée)
- **Port exposé**: 8002
- **Volumes**: artifacts/, data/, mlruns/

### Docker Compose

Le fichier `docker-compose.yml` configure :
- **Service**: fraud-detection-api
- **Port mapping**: 8002:8002
- **Volumes**: Persistance des modèles et données
- **Health check**: Vérifie que l'API répond sur /health
- **Restart policy**: Redémarre automatiquement en cas d'erreur

## 🔧 Configuration

### Variables d'environnement

Vous pouvez ajouter des variables d'environnement dans `docker-compose.yml` :

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - MODEL_PATH=/app/artifacts/models/random_forest_model.pkl
  - THRESHOLD_PATH=/app/artifacts/models/best_threshold.txt
```

### Volumes

Les volumes permettent de :
- **artifacts/**: Sauvegarder les modèles entraînés
- **data/**: Accéder aux données
- **mlruns/**: Conserver l'historique MLflow

## 🧪 Tester l'API

Une fois l'API lancée :

```bash
# Health check
curl http://localhost:8002/health

# Test de prédiction
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 150.50,
    "num_transactions_24h": 3,
    "account_age_days": 365,
    "is_foreign_transaction": 0,
    "country_risk": "low",
    "device_type": "mobile"
  }'

# Interface web
open http://localhost:8002
```

## 📊 Commandes utiles

```bash
# Reconstruire l'image
docker-compose build --no-cache

# Voir les logs en temps réel
docker-compose logs -f fraud-detection-api

# Entrer dans le conteneur
docker-compose exec fraud-detection-api bash

# Voir les ressources utilisées
docker stats fraud-detection-api

# Nettoyer tout
docker-compose down -v
docker system prune -a
```

## 🔍 Health Check

Le health check vérifie automatiquement que l'API est opérationnelle :
- **Intervalle**: 30 secondes
- **Timeout**: 10 secondes
- **Retries**: 3 tentatives
- **Start period**: 40 secondes

Si le health check échoue, le conteneur sera marqué comme "unhealthy".

## 🐛 Troubleshooting

### Le conteneur ne démarre pas

```bash
# Vérifier les logs
docker-compose logs

# Vérifier si le port 8002 est déjà utilisé
lsof -i :8002
```

### Le modèle n'est pas chargé

Assurez-vous d'avoir entraîné le modèle avant de lancer Docker :

```bash
# Entraîner le modèle
python run_pipeline.py --trials 30

# Vérifier que le modèle existe
ls -la artifacts/models/
```

### Problème de permissions

Si vous avez des problèmes de permissions avec les volumes :

```bash
# Changer les permissions
chmod -R 755 artifacts/ data/ mlruns/
```

## 🚢 Déploiement en production

### Avec Docker Hub

1. **Push l'image**:
```bash
docker tag fraud-detection-api username/fraud-detection:latest
docker push username/fraud-detection:latest
```

2. **Pull et run sur le serveur**:
```bash
docker pull username/fraud-detection:latest
docker run -d -p 8002:8002 username/fraud-detection:latest
```

### Avec GitHub Actions

Le workflow `.github/workflows/ci-cd.yml` automatise :
1. Tests avec pytest
2. Build de l'image Docker
3. Push sur Docker Hub
4. Tag avec le SHA du commit

Configurez les secrets dans GitHub :
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## 🔐 Sécurité

Pour la production, considérez :
- Utiliser un reverse proxy (Nginx, Traefik)
- Activer HTTPS
- Limiter l'accès avec un firewall
- Utiliser des secrets pour les credentials
- Scanner l'image avec `docker scan`

## 📈 Monitoring

Pour monitorer l'API en production :

```bash
# Métriques du conteneur
docker stats fraud-detection-api

# Logs en continu
docker-compose logs -f --tail=100

# Health check manuel
curl http://localhost:8002/health
```

## ✅ Checklist de déploiement

- [ ] Modèle entraîné et sauvegardé dans artifacts/
- [ ] Requirements.txt à jour
- [ ] Tests passent localement
- [ ] Docker build réussit
- [ ] Health check répond
- [ ] API répond sur /predict
- [ ] Interface web accessible
- [ ] Volumes correctement montés
- [ ] Secrets Docker Hub configurés (pour CI/CD)
