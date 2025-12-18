"""
FastAPI application pour la détection de fraude.
API simple et propre avec interface web de visualisation.
"""

import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pickle
from typing import List
import numpy as np

from api.schemas import (
    TransactionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthCheck
)


# Initialiser FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude avec visualisation web",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Variables globales
model = None
threshold = 0.5
feature_names = None  # Sera chargé depuis le modèle


def load_model_and_threshold():
    """Charge le modèle et le threshold au démarrage."""
    global model, threshold, feature_names

    try:
        # Charger le modèle
        model_path = project_root / "artifacts/models/random_forest_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Récupérer les noms de features du modèle
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
                print(f"✓ Modèle chargé avec {len(feature_names)} features")
            else:
                print("⚠ Modèle chargé mais features names non disponibles")
                return False
        else:
            print("⚠ Modèle non trouvé - Exécutez d'abord: python run_pipeline.py")
            return False

        # Charger le threshold
        threshold_path = project_root / "artifacts/models/best_threshold.txt"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
            print(f"✓ Threshold chargé: {threshold:.4f}")
        else:
            print(f"⚠ Threshold non trouvé - Utilisation par défaut: {threshold}")

        return True

    except Exception as e:
        print(f"❌ Erreur au chargement: {e}")
        return False


@app.on_event("startup")
def startup_event():
    """Initialiser le modèle au démarrage."""
    success = load_model_and_threshold()
    if not success:
        print("⚠ API démarrée mais modèle non chargé")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil avec interface web de visualisation."""
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fraud Detection API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 40px;
            }
            .header h1 {
                font-size: 3em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            input, select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
            }
            .btn:active {
                transform: translateY(0);
            }
            .result {
                display: none;
                margin-top: 30px;
                padding: 25px;
                border-radius: 12px;
                text-align: center;
            }
            .result.show {
                display: block;
            }
            .result.fraud {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                color: white;
            }
            .result.no-fraud {
                background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
                color: white;
            }
            .result h2 {
                font-size: 2em;
                margin-bottom: 15px;
            }
            .result .probability {
                font-size: 3em;
                font-weight: bold;
                margin: 20px 0;
            }
            .result .details {
                font-size: 1.1em;
                opacity: 0.95;
                margin-top: 15px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .links {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 30px;
            }
            .links a {
                color: white;
                text-decoration: none;
                padding: 12px 30px;
                background: rgba(255,255,255,0.2);
                border-radius: 8px;
                font-weight: 600;
                transition: background 0.3s;
            }
            .links a:hover {
                background: rgba(255,255,255,0.3);
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Fraud Detection</h1>
                <p>Détection de fraude bancaire par Machine Learning</p>
            </div>

            <div class="card">
                <h2 style="margin-bottom: 25px; color: #333;">Analyser une transaction</h2>
                <form id="fraudForm">
                    <div class="grid">
                        <div class="form-group">
                            <label for="transaction_amount">Montant (€)</label>
                            <input type="number" step="0.01" id="transaction_amount" required value="150.50">
                        </div>
                        <div class="form-group">
                            <label for="num_transactions_24h">Transactions 24h</label>
                            <input type="number" id="num_transactions_24h" required value="3">
                        </div>
                    </div>

                    <div class="grid">
                        <div class="form-group">
                            <label for="account_age_days">Âge du compte (jours)</label>
                            <input type="number" id="account_age_days" required value="365">
                        </div>
                        <div class="form-group">
                            <label for="is_foreign_transaction">Transaction étrangère</label>
                            <select id="is_foreign_transaction" required>
                                <option value="0">Non</option>
                                <option value="1">Oui</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid">
                        <div class="form-group">
                            <label for="country_risk">Risque pays</label>
                            <select id="country_risk" required>
                                <option value="low">Faible</option>
                                <option value="medium">Moyen</option>
                                <option value="high">Élevé</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="device_type">Type d'appareil</label>
                            <select id="device_type" required>
                                <option value="desktop">Desktop</option>
                                <option value="mobile">Mobile</option>
                                <option value="tablet">Tablet</option>
                            </select>
                        </div>
                    </div>

                    <button type="submit" class="btn">Analyser la transaction</button>
                </form>

                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p style="margin-top: 15px;">Analyse en cours...</p>
                </div>

                <div class="result" id="result">
                    <h2 id="resultTitle"></h2>
                    <div class="probability" id="probability"></div>
                    <div class="details" id="details"></div>
                </div>
            </div>

            <div class="links">
                <a href="/docs" target="_blank">📚 Documentation API</a>
                <a href="/health" target="_blank">💚 Health Check</a>
            </div>
        </div>

        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                // Récupérer les valeurs
                const data = {
                    transaction_amount: parseFloat(document.getElementById('transaction_amount').value),
                    num_transactions_24h: parseInt(document.getElementById('num_transactions_24h').value),
                    account_age_days: parseInt(document.getElementById('account_age_days').value),
                    is_foreign_transaction: parseInt(document.getElementById('is_foreign_transaction').value),
                    country_risk: document.getElementById('country_risk').value,
                    device_type: document.getElementById('device_type').value
                };

                // Afficher le loading
                document.getElementById('loading').classList.add('show');
                document.getElementById('result').classList.remove('show');

                try {
                    // Appeler l'API
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();

                    // Masquer le loading
                    document.getElementById('loading').classList.remove('show');

                    // Afficher le résultat
                    const resultDiv = document.getElementById('result');
                    const probability = result.fraud_probability * 100;

                    if (result.fraud_prediction === 1) {
                        resultDiv.className = 'result fraud show';
                        document.getElementById('resultTitle').textContent = '⚠️ FRAUDE DÉTECTÉE';
                        document.getElementById('details').textContent =
                            `Cette transaction présente un niveau de risque ${result.risk_level.toLowerCase()}. Une vérification manuelle est recommandée.`;
                    } else {
                        resultDiv.className = 'result no-fraud show';
                        document.getElementById('resultTitle').textContent = '✅ TRANSACTION LÉGITIME';
                        document.getElementById('details').textContent =
                            `Cette transaction semble normale avec un niveau de risque ${result.risk_level.toLowerCase()}.`;
                    }

                    document.getElementById('probability').textContent = `${probability.toFixed(1)}%`;

                } catch (error) {
                    document.getElementById('loading').classList.remove('show');
                    alert('Erreur lors de l\'analyse: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint.
    Vérifie que l'API et le modèle sont opérationnels.
    """
    return HealthCheck(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        threshold=threshold,
        version="1.0.0"
    )


def prepare_features(transaction: TransactionInput) -> np.ndarray:
    """
    Prépare les features pour la prédiction.
    Retourne un array numpy dans le bon ordre.
    """
    # Créer le DataFrame de base
    data = {
        'is_foreign_transaction': [transaction.is_foreign_transaction],
        'account_age_days': [transaction.account_age_days],
        'num_transactions_24h': [transaction.num_transactions_24h],
        'transaction_amount': [transaction.transaction_amount],
    }

    # One-Hot Encoding pour country_risk
    data['country_risk_high'] = [1 if transaction.country_risk == 'high' else 0]
    data['country_risk_low'] = [1 if transaction.country_risk == 'low' else 0]
    data['country_risk_medium'] = [1 if transaction.country_risk == 'medium' else 0]

    # One-Hot Encoding pour device_type
    data['device_type_desktop'] = [1 if transaction.device_type == 'desktop' else 0]
    data['device_type_mobile'] = [1 if transaction.device_type == 'mobile' else 0]
    data['device_type_tablet'] = [1 if transaction.device_type == 'tablet' else 0]

    # Créer le DataFrame avec les colonnes dans le bon ordre
    df = pd.DataFrame(data, columns=feature_names)

    return df


def get_risk_level(probability: float) -> str:
    """Détermine le niveau de risque selon la probabilité."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """
    Prédit si une transaction est frauduleuse.

    Args:
        transaction: Données de la transaction

    Returns:
        Probabilité de fraude et prédiction binaire
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord: python run_pipeline.py"
        )

    try:
        # Préparer les features
        X = prepare_features(transaction)

        # Prédiction
        y_proba = model.predict_proba(X)[:, 1][0]
        y_pred = int(y_proba >= threshold)

        # Niveau de risque
        risk_level = get_risk_level(y_proba)

        return PredictionOutput(
            fraud_probability=round(float(y_proba), 4),
            fraud_prediction=y_pred,
            fraud_prediction_label="Fraud" if y_pred == 1 else "Legitimate",
            risk_level=risk_level,
            threshold=threshold
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchPredictionInput):
    """
    Prédit la fraude pour plusieurs transactions.

    Args:
        batch: Liste de transactions

    Returns:
        Prédictions pour chaque transaction + statistiques
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Exécutez d'abord: python run_pipeline.py"
        )

    try:
        predictions = []

        for transaction in batch.transactions:
            # Préparer les features
            X = prepare_features(transaction)

            # Prédiction
            y_proba = model.predict_proba(X)[:, 1][0]
            y_pred = int(y_proba >= threshold)

            predictions.append(
                PredictionOutput(
                    fraud_probability=round(float(y_proba), 4),
                    fraud_prediction=y_pred,
                    fraud_prediction_label="Fraud" if y_pred == 1 else "Legitimate",
                    risk_level=get_risk_level(y_proba),
                    threshold=threshold
                )
            )

        # Statistiques
        total = len(predictions)
        frauds = sum(1 for p in predictions if p.fraud_prediction == 1)
        fraud_rate = frauds / total if total > 0 else 0

        return BatchPredictionOutput(
            predictions=predictions,
            total_transactions=total,
            predicted_frauds=frauds,
            fraud_rate=round(fraud_rate, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction batch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
