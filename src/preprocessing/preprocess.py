"""
Module de prétraitement des données.

Ce module nettoie les données et les prépare pour le feature engineering.
Pour ce dataset, les données sont déjà propres, donc ce module est minimal.
"""

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données brutes.

    Pour ce dataset, les données sont déjà propres :
    - Pas de valeurs manquantes
    - Pas de doublons
    - Types de données corrects

    Args:
        df: DataFrame brut

    Returns:
        DataFrame prétraité (identique au brut pour ce cas)
    """
    # Créer une copie pour ne pas modifier l'original
    df_clean = df.copy()

    # Vérifier qu'il n'y a pas de valeurs manquantes
    n_missing = df_clean.isnull().sum().sum()
    if n_missing > 0:
        print(f"⚠️ {n_missing} valeurs manquantes détectées")
        # Supprimer les lignes avec valeurs manquantes
        df_clean = df_clean.dropna()
        print(f"✓ Valeurs manquantes supprimées")
    else:
        print("✓ Aucune valeur manquante")

    # Vérifier les doublons
    n_duplicates = df_clean.duplicated().sum()
    if n_duplicates > 0:
        print(f"⚠️ {n_duplicates} doublons détectés")
        df_clean = df_clean.drop_duplicates()
        print(f"✓ Doublons supprimés")
    else:
        print("✓ Aucun doublon")

    print(f"✓ Données prétraitées : {df_clean.shape[0]} lignes restantes")

    return df_clean


def validate_data(df: pd.DataFrame) -> bool:
    """
    Valide que les données sont correctes.

    Args:
        df: DataFrame à valider

    Returns:
        True si les données sont valides, False sinon
    """
    # Vérifier que la colonne fraud existe
    if 'fraud' not in df.columns:
        print("❌ Colonne 'fraud' manquante")
        return False

    # Vérifier que fraud ne contient que 0 et 1
    unique_values = df['fraud'].unique()
    if not all(v in [0, 1] for v in unique_values):
        print(f"❌ La colonne 'fraud' contient des valeurs invalides : {unique_values}")
        return False

    # Vérifier qu'il n'y a pas de valeurs manquantes
    if df.isnull().sum().sum() > 0:
        print("❌ Valeurs manquantes détectées")
        return False

    print("✓ Données valides")
    return True


if __name__ == "__main__":
    # Test du module
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    df_clean = preprocess_data(df)
    is_valid = validate_data(df_clean)

    print(f"\n✓ Test terminé - Données valides : {is_valid}")
