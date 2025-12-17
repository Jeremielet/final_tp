"""
Module de feature engineering.

Ce module sélectionne les features importantes et encode les variables catégorielles.
"""

import pandas as pd
import numpy as np


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne uniquement les features importantes basées sur l'analyse.

    Features gardées (6 + cible) :
    - is_foreign_transaction
    - account_age_days
    - country_risk
    - num_transactions_24h
    - transaction_amount
    - device_type
    - fraud (cible)

    Args:
        df: DataFrame complet

    Returns:
        DataFrame avec uniquement les features sélectionnées
    """
    # Liste des features à garder
    selected_cols = [
        'is_foreign_transaction',  # Impact +78.5%
        'account_age_days',        # Impact -23.9%
        'country_risk',            # Impact massif
        'num_transactions_24h',    # Impact +15.1%
        'transaction_amount',      # Impact +15.3%
        'device_type',             # Impact modéré
        'fraud'                    # Variable cible
    ]

    # Sélectionner les colonnes
    df_selected = df[selected_cols].copy()

    print(f"✓ Features sélectionnées : {len(selected_cols) - 1} features + 1 cible")
    print(f"  Features : {[c for c in selected_cols if c != 'fraud']}")

    return df_selected


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles avec One-Hot Encoding.

    Variables à encoder :
    - country_risk : 3 catégories (low, medium, high)
    - device_type : 3 catégories (desktop, mobile, tablet)

    Args:
        df: DataFrame avec features sélectionnées

    Returns:
        DataFrame avec variables encodées
    """
    # Créer une copie
    df_encoded = df.copy()

    # One-Hot Encoding pour country_risk
    if 'country_risk' in df_encoded.columns:
        country_dummies = pd.get_dummies(
            df_encoded['country_risk'],
            prefix='country_risk',
            drop_first=False  # Garder toutes les catégories
        )
        df_encoded = pd.concat([df_encoded, country_dummies], axis=1)
        df_encoded = df_encoded.drop('country_risk', axis=1)
        print(f"✓ country_risk encodé : {list(country_dummies.columns)}")

    # One-Hot Encoding pour device_type
    if 'device_type' in df_encoded.columns:
        device_dummies = pd.get_dummies(
            df_encoded['device_type'],
            prefix='device_type',
            drop_first=False  # Garder toutes les catégories
        )
        df_encoded = pd.concat([df_encoded, device_dummies], axis=1)
        df_encoded = df_encoded.drop('device_type', axis=1)
        print(f"✓ device_type encodé : {list(device_dummies.columns)}")

    print(f"✓ Encodage terminé : {df_encoded.shape[1]} colonnes finales")

    return df_encoded


def split_features_target(df: pd.DataFrame) -> tuple:
    """
    Sépare les features (X) de la cible (y).

    Args:
        df: DataFrame avec features et cible

    Returns:
        Tuple (X, y) où X = features, y = cible
    """
    # Séparer X et y
    X = df.drop('fraud', axis=1)
    y = df['fraud'].values

    print(f"✓ Données séparées :")
    print(f"  X : {X.shape[0]} lignes × {X.shape[1]} features")
    print(f"  y : {y.shape[0]} valeurs")
    print(f"  Fraudes : {y.sum()} ({y.sum()/len(y)*100:.2f}%)")

    return X, y


def build_features(df: pd.DataFrame) -> tuple:
    """
    Pipeline complet de feature engineering.

    Args:
        df: DataFrame brut prétraité

    Returns:
        Tuple (X, y) prêt pour l'entraînement
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    # Étape 1 : Sélectionner les features
    df_selected = select_features(df)

    # Étape 2 : Encoder les catégorielles
    df_encoded = encode_categorical(df_selected)

    # Étape 3 : Séparer X et y
    X, y = split_features_target(df_encoded)

    print("="*80)
    print("✓ Feature engineering terminé")
    print("="*80 + "\n")

    return X, y


if __name__ == "__main__":
    # Test du module
    from src.data.load_data import load_raw_data
    from src.preprocessing.preprocess import preprocess_data

    df = load_raw_data()
    df_clean = preprocess_data(df)
    X, y = build_features(df_clean)

    print(f"\n✓ Test terminé")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  X columns : {X.columns.tolist()}")
