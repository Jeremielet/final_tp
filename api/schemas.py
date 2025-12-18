"""
Schémas Pydantic pour l'API de détection de fraude.
Définit les structures de données pour les requêtes et réponses.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class TransactionInput(BaseModel):
    """
    Données d'entrée pour une transaction.
    """
    transaction_amount: float = Field(..., description="Montant de la transaction", example=150.50)
    num_transactions_24h: int = Field(..., description="Nombre de transactions dans les 24h", example=3)
    account_age_days: int = Field(..., description="Âge du compte en jours", example=365)
    is_foreign_transaction: int = Field(..., description="Transaction étrangère (0=Non, 1=Oui)", example=0)
    country_risk: Literal['low', 'medium', 'high'] = Field(..., description="Niveau de risque du pays", example="low")
    device_type: Literal['desktop', 'mobile', 'tablet'] = Field(..., description="Type d'appareil", example="mobile")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amount": 150.50,
                "num_transactions_24h": 3,
                "account_age_days": 365,
                "is_foreign_transaction": 0,
                "country_risk": "low",
                "device_type": "mobile"
            }
        }


class PredictionOutput(BaseModel):
    """
    Résultat de prédiction pour une transaction.
    """
    fraud_probability: float = Field(..., description="Probabilité de fraude (0-1)")
    fraud_prediction: int = Field(..., description="Prédiction binaire (0=Légitime, 1=Fraude)")
    fraud_prediction_label: str = Field(..., description="Label de la prédiction")
    risk_level: str = Field(..., description="Niveau de risque (Low/Medium/High)")
    threshold: float = Field(..., description="Seuil de décision utilisé")

    class Config:
        json_schema_extra = {
            "example": {
                "fraud_probability": 0.1234,
                "fraud_prediction": 0,
                "fraud_prediction_label": "Legitimate",
                "risk_level": "Low",
                "threshold": 0.5
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Liste de transactions pour prédiction batch.
    """
    transactions: List[TransactionInput] = Field(..., description="Liste des transactions à analyser")

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "transaction_amount": 150.50,
                        "num_transactions_24h": 3,
                        "account_age_days": 365,
                        "is_foreign_transaction": 0,
                        "country_risk": "low",
                        "device_type": "mobile"
                    },
                    {
                        "transaction_amount": 2500.00,
                        "num_transactions_24h": 15,
                        "account_age_days": 30,
                        "is_foreign_transaction": 1,
                        "country_risk": "high",
                        "device_type": "desktop"
                    }
                ]
            }
        }


class BatchPredictionOutput(BaseModel):
    """
    Résultats de prédiction batch avec statistiques.
    """
    predictions: List[PredictionOutput] = Field(..., description="Liste des prédictions")
    total_transactions: int = Field(..., description="Nombre total de transactions")
    predicted_frauds: int = Field(..., description="Nombre de fraudes détectées")
    fraud_rate: float = Field(..., description="Taux de fraude détecté")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "fraud_probability": 0.1234,
                        "fraud_prediction": 0,
                        "fraud_prediction_label": "Legitimate",
                        "risk_level": "Low",
                        "threshold": 0.5
                    }
                ],
                "total_transactions": 10,
                "predicted_frauds": 2,
                "fraud_rate": 0.2
            }
        }


class HealthCheck(BaseModel):
    """
    Health check de l'API.
    """
    status: str = Field(..., description="Statut de l'API (healthy/degraded)")
    model_loaded: bool = Field(..., description="Modèle chargé ou non")
    threshold: float = Field(..., description="Seuil de décision actuel")
    version: str = Field(..., description="Version de l'API")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "threshold": 0.5,
                "version": "1.0.0"
            }
        }
