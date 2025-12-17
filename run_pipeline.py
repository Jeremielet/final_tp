"""
Pipeline principal d'entraînement du modèle de détection de fraude.

Ce script exécute le pipeline complet :
1. Chargement des données
2. Prétraitement
3. Feature engineering
4. Entraînement avec Optuna + MLflow
5. Sauvegarde du modèle

Usage:
    python run_pipeline.py
"""

import argparse
from pathlib import Path

# Imports des modules du projet
from src.data.load_data import load_raw_data, get_data_info
from src.preprocessing.preprocess import preprocess_data, validate_data
from src.features.build_features import build_features
from src.models.train import train_with_mlflow


def main(data_path: str = "data/raw/fraud_synth_10000.csv",
         n_trials: int = 30,
         experiment_name: str = "Fraud Detection"):
    """
    Exécute le pipeline complet d'entraînement.

    Args:
        data_path: Chemin vers les données brutes
        n_trials: Nombre d'essais pour Optuna
        experiment_name: Nom de l'expérience MLflow
    """
    print("="*80)
    print("🚀 PIPELINE DE DÉTECTION DE FRAUDE")
    print("="*80)
    print(f"Configuration :")
    print(f"  - Data: {data_path}")
    print(f"  - Optuna trials: {n_trials}")
    print(f"  - MLflow experiment: {experiment_name}")
    print("="*80 + "\n")

    # ========================================================================
    # ÉTAPE 1 : Chargement des données
    # ========================================================================
    print("📂 ÉTAPE 1/5 : Chargement des données")
    print("-" * 80)

    df = load_raw_data(data_path)
    info = get_data_info(df)

    print(f"\n📊 Informations du dataset :")
    print(f"  - Lignes : {info['n_rows']:,}")
    print(f"  - Colonnes : {info['n_columns']}")
    print(f"  - Fraudes : {info['fraud_count']} ({info['fraud_percentage']:.2f}%)")
    print(f"  - Valeurs manquantes : {info['n_missing']}")
    print()

    # ========================================================================
    # ÉTAPE 2 : Prétraitement
    # ========================================================================
    print("🧹 ÉTAPE 2/5 : Prétraitement des données")
    print("-" * 80)

    df_clean = preprocess_data(df)
    is_valid = validate_data(df_clean)

    if not is_valid:
        print("❌ Données invalides - Arrêt du pipeline")
        return
    print()

    # ========================================================================
    # ÉTAPE 3 : Feature Engineering
    # ========================================================================
    print("⚙️ ÉTAPE 3/5 : Feature Engineering")
    print("-" * 80)

    X, y = build_features(df_clean)
    print()

    # ========================================================================
    # ÉTAPE 4 : Entraînement avec Optuna + MLflow
    # ========================================================================
    print("🤖 ÉTAPE 4/5 : Entraînement du modèle")
    print("-" * 80)

    model, metrics = train_with_mlflow(
        X, y,
        experiment_name=experiment_name,
        n_trials=n_trials
    )
    print()

    # ========================================================================
    # ÉTAPE 5 : Résumé final
    # ========================================================================
    print("=" * 80)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 80)
    print(f"\n📊 Résultats finaux :")
    print(f"  - Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  - Precision : {metrics['precision']:.4f}")
    print(f"  - Recall    : {metrics['recall']:.4f} ⭐")
    print(f"  - F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC   : {metrics['roc_auc']:.4f}")

    print(f"\n💾 Modèle sauvegardé :")
    print(f"  - Fichier : artifacts/models/random_forest_model.pkl")

    print(f"\n📈 MLflow :")
    print(f"  - Experiment : {experiment_name}")
    print(f"  - Lancer l'UI : mlflow ui")
    print(f"  - URL : http://localhost:5000")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Pipeline de détection de fraude avec Random Forest"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/fraud_synth_10000.csv",
        help="Chemin vers les données brutes (default: data/raw/fraud_synth_10000.csv)"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Nombre d'essais Optuna (default: 30)"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="Fraud Detection",
        help="Nom de l'expérience MLflow (default: 'Fraud Detection')"
    )

    args = parser.parse_args()

    # Exécuter le pipeline
    main(
        data_path=args.data,
        n_trials=args.trials,
        experiment_name=args.experiment
    )
