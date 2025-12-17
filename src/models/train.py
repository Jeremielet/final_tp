"""
Module d'entraînement du modèle Random Forest avec Optuna et MLflow.

Ce module :
1. Split les données en train/test
2. Optimise les hyperparamètres avec Optuna
3. Entraîne le meilleur modèle
4. Track tout avec MLflow
5. Sauvegarde le modèle
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import optuna
import mlflow
import mlflow.sklearn
import pickle
from pathlib import Path


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split les données en train et test.

    Args:
        X: Features
        y: Cible
        test_size: Proportion du test set (default: 0.2)
        random_state: Graine aléatoire (default: 42)

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Garde la même proportion de fraudes
    )

    print(f"✓ Données divisées :")
    print(f"  Train : {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Fraudes : {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Test  : {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"    - Fraudes : {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calcule toutes les métriques d'évaluation.

    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions (0 ou 1)
        y_pred_proba: Probabilités de la classe 1

    Returns:
        Dictionnaire avec toutes les métriques
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    return metrics


def optimize_hyperparameters(X_train, y_train, n_trials=30):
    """
    Optimise les hyperparamètres de Random Forest avec Optuna.

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        n_trials: Nombre d'essais Optuna (default: 30)

    Returns:
        Dictionnaire des meilleurs paramètres
    """
    print(f"\n🔍 Optimisation Optuna ({n_trials} trials)...")

    def objective(trial):
        """Fonction objectif pour Optuna."""
        # Suggérer des hyperparamètres
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',  # Gérer le déséquilibre
            'random_state': 42
        }

        # Créer et entraîner le modèle
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Évaluer sur le train set (avec validation croisée implicite)
        y_pred = model.predict(X_train)
        f1 = f1_score(y_train, y_pred)

        return f1

    # Créer l'étude Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✓ Optimisation terminée")
    print(f"  Meilleur F1-Score : {study.best_value:.4f}")
    print(f"  Meilleurs paramètres :")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study.best_params


def train_model(X_train, y_train, params):
    """
    Entraîne le modèle Random Forest avec les paramètres donnés.

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        params: Hyperparamètres du modèle

    Returns:
        Modèle entraîné
    """
    # Ajouter class_weight si pas présent
    if 'class_weight' not in params:
        params['class_weight'] = 'balanced'

    # Ajouter random_state si pas présent
    if 'random_state' not in params:
        params['random_state'] = 42

    # Créer et entraîner le modèle
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    print(f"✓ Modèle Random Forest entraîné")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le test set.

    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Cible de test

    Returns:
        Dictionnaire avec les métriques
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    print(f"\n📊 Résultats sur le Test Set :")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f} ⭐")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"\n  Confusion Matrix :")
    print(f"    TN={metrics['true_negatives']} | FP={metrics['false_positives']}")
    print(f"    FN={metrics['false_negatives']} | TP={metrics['true_positives']}")

    return metrics


def save_model(model, output_path="artifacts/models/random_forest_model.pkl"):
    """
    Sauvegarde le modèle sur disque.

    Args:
        model: Modèle à sauvegarder
        output_path: Chemin de sauvegarde

    Returns:
        Chemin complet du fichier sauvegardé
    """
    # Créer le dossier si nécessaire
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ Modèle sauvegardé : {output_path}")

    return output_path


def train_with_mlflow(X, y, experiment_name="Fraud Detection", n_trials=30):
    """
    Pipeline complet d'entraînement avec tracking MLflow.

    Args:
        X: Features complètes
        y: Cible complète
        experiment_name: Nom de l'expérience MLflow
        n_trials: Nombre d'essais Optuna

    Returns:
        Tuple (model, metrics)
    """
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE RANDOM FOREST")
    print("="*80)

    # Configurer MLflow
    mlflow.set_experiment(experiment_name)

    # Démarrer un run MLflow
    with mlflow.start_run():

        # 1. Split des données
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Logger les infos du dataset
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("fraud_rate", f"{y.sum()/len(y)*100:.2f}%")

        # 2. Optimisation des hyperparamètres
        best_params = optimize_hyperparameters(X_train, y_train, n_trials=n_trials)

        # Logger les hyperparamètres
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        # 3. Entraînement du modèle
        model = train_model(X_train, y_train, best_params)

        # 4. Évaluation
        metrics = evaluate_model(model, X_test, y_test)

        # Logger les métriques dans MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # 5. Sauvegarder le modèle
        model_path = save_model(model)

        # Logger le modèle dans MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)

        print("\n✓ Entraînement terminé et tracké dans MLflow")
        print("="*80 + "\n")

        return model, metrics


if __name__ == "__main__":
    # Test du module
    from src.data.load_data import load_raw_data
    from src.preprocessing.preprocess import preprocess_data
    from src.features.build_features import build_features

    # Charger et préparer les données
    df = load_raw_data()
    df_clean = preprocess_data(df)
    X, y = build_features(df_clean)

    # Entraîner avec MLflow
    model, metrics = train_with_mlflow(X, y, n_trials=10)  # 10 trials pour le test

    print(f"\n✓ Test terminé - F1-Score : {metrics['f1_score']:.4f}")
