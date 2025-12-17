"""
Module de chargement des données.

Ce module charge le fichier CSV brut et retourne un DataFrame pandas.
"""

import pandas as pd
from pathlib import Path


def load_raw_data(data_path: str = "data/raw/fraud_synth_10000.csv") -> pd.DataFrame:
    """
    Charge les données brutes depuis le fichier CSV.

    Args:
        data_path: Chemin vers le fichier CSV

    Returns:
        DataFrame pandas avec les données brutes

    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    # Vérifier que le fichier existe
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Le fichier {data_path} n'existe pas")

    # Charger le CSV
    df = pd.read_csv(data_path)

    print(f"✓ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"  Colonnes : {df.columns.tolist()}")

    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Retourne des informations sur le dataset.

    Args:
        df: DataFrame à analyser

    Returns:
        Dictionnaire avec les informations du dataset
    """
    info = {
        'n_rows': len(df),
        'n_columns': df.shape[1],
        'n_missing': df.isnull().sum().sum(),
        'fraud_count': df['fraud'].sum() if 'fraud' in df.columns else None,
        'fraud_percentage': (df['fraud'].sum() / len(df) * 100) if 'fraud' in df.columns else None
    }

    return info


if __name__ == "__main__":
    # Test du module
    df = load_raw_data()
    info = get_data_info(df)

    print("\n📊 Informations du dataset :")
    for key, value in info.items():
        if value is not None:
            print(f"  {key}: {value}")
