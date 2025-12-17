# 🔍 Fraud Detection - Final TP

Projet de détection de fraude utilisant des techniques de Machine Learning.

## 📊 Dataset

**fraud_synth_10000.csv** : Dataset synthétique de transactions avec détection de fraude

Variables disponibles :
- `transaction_amount` : Montant de la transaction
- `transaction_hour` : Heure de la transaction
- `num_transactions_24h` : Nombre de transactions dans les dernières 24h
- `account_age_days` : Âge du compte en jours
- `avg_amount_30d` : Montant moyen sur 30 jours
- `country_risk` : Niveau de risque du pays
- `device_type` : Type d'appareil
- `is_foreign_transaction` : Transaction étrangère (0/1)
- `fraud` : Variable cible (0=Normal, 1=Fraude)

## 📁 Structure du Projet

```
final_tp/
├── data/
│   ├── raw/              # Données brutes
│   ├── processed/        # Données prétraitées
│   └── external/         # Données externes
├── notebooks/            # Jupyter notebooks
│   ├── 01_load_data.ipynb
│   └── 02_analyze_data.ipynb
├── src/                  # Code source
│   ├── data/            # Chargement des données
│   ├── preprocessing/   # Prétraitement
│   ├── features/        # Feature engineering
│   ├── models/          # Modèles ML
│   └── utils/           # Utilitaires
├── tests/               # Tests unitaires
├── scripts/             # Scripts d'exécution
├── configs/             # Configurations
├── artifacts/           # Modèles et résultats
├── mlruns/              # Tracking MLflow
└── docs/                # Documentation

```

## 🚀 Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Jeremielet/final_tp.git
cd final_tp
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📓 Utilisation des Notebooks

### Lancer Jupyter

```bash
jupyter notebook
```

### Notebooks disponibles

1. **01_load_data.ipynb** : Chargement et première exploration des données
2. **02_analyze_data.ipynb** : Analyse exploratoire détaillée avec visualisations

## 🔧 Workflow du Projet

1. **Exploration** : Notebooks 01 et 02
2. **Prétraitement** : Nettoyage et préparation des données
3. **Feature Engineering** : Création de nouvelles features
4. **Modélisation** : Entraînement de modèles ML
5. **Évaluation** : Mesure des performances
6. **Déploiement** : API FastAPI + interface Gradio

## 📦 Technologies Utilisées

- **Data Science** : pandas, numpy, scikit-learn
- **Visualisation** : matplotlib, seaborn
- **Machine Learning** : XGBoost, imbalanced-learn
- **MLOps** : MLflow
- **API** : FastAPI, Gradio
- **Testing** : pytest

## 🎯 Objectif

Développer un modèle de Machine Learning capable de détecter les transactions frauduleuses avec une haute précision tout en minimisant les faux positifs.

## 👨‍💻 Auteur

Jérémie Letarnec

## 📝 Licence

Projet à but éducatif - 2024
