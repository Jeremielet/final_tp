"""
Tests rapides pour l'API de détection de fraude.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.main import app

# Client de test
client = TestClient(app)


def test_health_check():
    """Test du health check."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "threshold" in data


def test_predict_legitimate_transaction():
    """Test prédiction transaction légitime."""
    transaction = {
        "transaction_amount": 150.50,
        "num_transactions_24h": 3,
        "account_age_days": 365,
        "is_foreign_transaction": 0,
        "country_risk": "low",
        "device_type": "mobile"
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    data = response.json()
    assert "fraud_probability" in data
    assert "fraud_prediction" in data
    assert "risk_level" in data
    assert data["fraud_probability"] >= 0
    assert data["fraud_probability"] <= 1


def test_predict_suspicious_transaction():
    """Test prédiction transaction suspecte."""
    transaction = {
        "transaction_amount": 5000.00,
        "num_transactions_24h": 25,
        "account_age_days": 10,
        "is_foreign_transaction": 1,
        "country_risk": "high",
        "device_type": "desktop"
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 200

    data = response.json()
    assert data["fraud_probability"] > 0.5  # Transaction suspecte
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_batch():
    """Test prédiction batch."""
    batch = {
        "transactions": [
            {
                "transaction_amount": 100.00,
                "num_transactions_24h": 2,
                "account_age_days": 500,
                "is_foreign_transaction": 0,
                "country_risk": "low",
                "device_type": "mobile"
            },
            {
                "transaction_amount": 3000.00,
                "num_transactions_24h": 20,
                "account_age_days": 5,
                "is_foreign_transaction": 1,
                "country_risk": "high",
                "device_type": "tablet"
            }
        ]
    }

    response = client.post("/predict_batch", json=batch)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert "total_transactions" in data
    assert "predicted_frauds" in data
    assert "fraud_rate" in data
    assert len(data["predictions"]) == 2
    assert data["total_transactions"] == 2


def test_invalid_country_risk():
    """Test avec country_risk invalide."""
    transaction = {
        "transaction_amount": 150.50,
        "num_transactions_24h": 3,
        "account_age_days": 365,
        "is_foreign_transaction": 0,
        "country_risk": "invalid",  # Invalide
        "device_type": "mobile"
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 422  # Validation error


def test_missing_field():
    """Test avec champ manquant."""
    transaction = {
        "transaction_amount": 150.50,
        # Champ manquant: num_transactions_24h
        "account_age_days": 365,
        "is_foreign_transaction": 0,
        "country_risk": "low",
        "device_type": "mobile"
    }

    response = client.post("/predict", json=transaction)
    assert response.status_code == 422  # Validation error


def test_root_endpoint():
    """Test de la page d'accueil."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
