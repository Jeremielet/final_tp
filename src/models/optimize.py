"""
Module d'optimisation avancée avec Optuna.

Ce module contient :
1. Cross-validation pour valider les hyperparamètres
2. Optimisation du threshold pour maximiser le Recall
3. Sélection des meilleurs paramètres via Optuna
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve
import optuna


def optimize_hyperparameters_with_cv(X_train, y_train, n_trials=30, cv_folds=5):
    """
    Optimise les hyperparamètres de Random Forest avec Optuna et Cross-Validation.

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        n_trials: Nombre d'essais Optuna (default: 30)
        cv_folds: Nombre de folds pour la cross-validation (default: 5)

    Returns:
        dict: Meilleurs paramètres trouvés
    """
    print(f"\n🔍 Optimisation avec Optuna ({n_trials} trials)")
    print(f"   Cross-validation : {cv_folds}-fold")
    print("-" * 80)

    def objective(trial):
        """
        Fonction objectif pour Optuna.
        Teste des hyperparamètres avec cross-validation.
        """
        # Suggérer des hyperparamètres
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'class_weight': 'balanced',  # Toujours balanced
            'random_state': 42,
            'n_jobs': -1
        }

        # Créer le modèle
        model = RandomForestClassifier(**params)

        # Cross-validation stratifiée (garde la même proportion de fraudes)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Calculer le F1-Score moyen sur tous les folds
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )

        # Retourner la moyenne des F1-Scores
        mean_f1 = scores.mean()

        return mean_f1

    # Créer l'étude Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Afficher les résultats
    print(f"\n✓ Optimisation terminée")
    print(f"  Meilleur F1-Score (CV) : {study.best_value:.4f}")
    print(f"\n  Meilleurs paramètres :")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study.best_params


def find_best_threshold(model, X_val, y_val, metric='recall', min_precision=0.5):
    """
    Trouve le meilleur threshold pour optimiser une métrique donnée.

    Pour la détection de fraude, on veut maximiser le Recall
    (détecter le maximum de fraudes) tout en gardant une Precision acceptable.

    Args:
        model: Modèle entraîné
        X_val: Features de validation
        y_val: Cible de validation
        metric: Métrique à optimiser ('recall', 'f1', 'precision')
        min_precision: Precision minimale à respecter (default: 0.5)

    Returns:
        tuple: (best_threshold, metrics_dict)
    """
    print(f"\n🎯 Optimisation du Threshold")
    print(f"   Métrique à maximiser : {metric}")
    print(f"   Precision minimale : {min_precision:.2f}")
    print("-" * 80)

    # Obtenir les probabilités de prédiction
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculer precision, recall pour différents thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

    # Calculer F1-Score pour chaque threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Trouver le meilleur threshold selon la métrique
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}

    for i, threshold in enumerate(thresholds):
        # Appliquer le threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculer les métriques
        recall = recalls[i]
        precision = precisions[i]
        f1 = f1_scores[i]

        # Vérifier la contrainte de precision minimale
        if precision < min_precision:
            continue

        # Sélectionner selon la métrique demandée
        if metric == 'recall':
            score = recall
        elif metric == 'f1':
            score = f1
        elif metric == 'precision':
            score = precision
        else:
            score = f1

        # Mettre à jour si meilleur
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

    print(f"\n✓ Meilleur threshold trouvé : {best_threshold:.4f}")
    print(f"  Precision : {best_metrics['precision']:.4f}")
    print(f"  Recall    : {best_metrics['recall']:.4f} ⭐")
    print(f"  F1-Score  : {best_metrics['f1_score']:.4f}")

    return best_threshold, best_metrics


def train_with_best_params(X_train, y_train, best_params):
    """
    Entraîne un modèle avec les meilleurs paramètres trouvés.

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        best_params: Meilleurs paramètres d'Optuna

    Returns:
        Modèle entraîné
    """
    print(f"\n🤖 Entraînement du modèle final")
    print("-" * 80)

    # Ajouter les paramètres fixes
    final_params = best_params.copy()
    final_params['class_weight'] = 'balanced'
    final_params['random_state'] = 42
    final_params['n_jobs'] = -1

    # Créer et entraîner le modèle
    model = RandomForestClassifier(**final_params)
    model.fit(X_train, y_train)

    print(f"✓ Modèle entraîné avec les meilleurs paramètres")

    return model


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Évalue un modèle avec un threshold personnalisé.

    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Cible de test
        threshold: Threshold de décision (default: 0.5)

    Returns:
        dict: Métriques d'évaluation
    """
    # Prédictions avec threshold personnalisé
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculer les métriques
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    return metrics


def optimize_and_train(X_train, y_train, X_val, y_val, n_trials=30, cv_folds=5):
    """
    Pipeline complet d'optimisation :
    1. Optimise les hyperparamètres avec Optuna + CV
    2. Entraîne le modèle avec les meilleurs paramètres
    3. Optimise le threshold sur le validation set
    4. Retourne le modèle final et le threshold optimal

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_val: Features de validation
        y_val: Cible de validation
        n_trials: Nombre d'essais Optuna
        cv_folds: Nombre de folds pour CV

    Returns:
        tuple: (model, best_threshold, best_params)
    """
    print("\n" + "="*80)
    print("OPTIMISATION COMPLÈTE")
    print("="*80)

    # 1. Optimiser les hyperparamètres avec CV
    best_params = optimize_hyperparameters_with_cv(
        X_train, y_train,
        n_trials=n_trials,
        cv_folds=cv_folds
    )

    # 2. Entraîner avec les meilleurs paramètres
    model = train_with_best_params(X_train, y_train, best_params)

    # 3. Optimiser le threshold
    best_threshold, threshold_metrics = find_best_threshold(
        model, X_val, y_val,
        metric='recall',  # Maximiser le Recall pour la détection de fraude
        min_precision=0.5  # Garder au moins 50% de precision
    )

    print("\n" + "="*80)
    print("✓ OPTIMISATION TERMINÉE")
    print("="*80)

    return model, best_threshold, best_params


if __name__ == "__main__":
    # Test du module
    from src.data.load_data import load_raw_data
    from src.preprocessing.preprocess import preprocess_data
    from src.features.build_features import build_features
    from sklearn.model_selection import train_test_split

    # Charger et préparer les données
    df = load_raw_data()
    df_clean = preprocess_data(df)
    X, y = build_features(df_clean)

    # Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # Optimiser et entraîner
    model, threshold, params = optimize_and_train(
        X_train, y_train,
        X_val, y_val,
        n_trials=10,  # 10 trials pour le test
        cv_folds=3
    )

    # Évaluer sur le test set
    metrics = evaluate_with_threshold(model, X_test, y_test, threshold)

    print(f"\n📊 Résultats sur le Test Set :")
    print(f"  Threshold : {metrics['threshold']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f} ⭐")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}")
