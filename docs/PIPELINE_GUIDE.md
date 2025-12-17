# Guide d'Utilisation du Pipeline

Ce guide explique comment utiliser le pipeline d'entraînement du modèle de détection de fraude.

## 🏗️ Architecture du Pipeline

Le pipeline est organisé en modules Python clairs et simples :

```
final_tp/
├── src/
│   ├── data/
│   │   └── load_data.py          # Chargement des données
│   ├── preprocessing/
│   │   └── preprocess.py         # Nettoyage des données
│   ├── features/
│   │   └── build_features.py     # Feature engineering
│   └── models/
│       └── train.py              # Entraînement + Optuna + MLflow
├── run_pipeline.py               # Script principal
└── configs/
    └── config.yaml               # Configuration
```

## 🚀 Utilisation

### 1. Exécution simple

```bash
python run_pipeline.py
```

### 2. Avec options personnalisées

```bash
# Changer le nombre d'essais Optuna
python run_pipeline.py --trials 50

# Changer le nom de l'expérience MLflow
python run_pipeline.py --experiment "Mon Experience"

# Utiliser un autre fichier de données
python run_pipeline.py --data data/raw/autre_fichier.csv
```

### 3. Toutes les options

```bash
python run_pipeline.py \
  --data data/raw/fraud_synth_10000.csv \
  --trials 30 \
  --experiment "Fraud Detection"
```

## 📊 Étapes du Pipeline

### Étape 1 : Chargement des données
**Fichier** : `src/data/load_data.py`

- Charge le fichier CSV brut
- Affiche les informations du dataset
- Vérifie que le fichier existe

```python
from src.data.load_data import load_raw_data

df = load_raw_data("data/raw/fraud_synth_10000.csv")
```

### Étape 2 : Prétraitement
**Fichier** : `src/preprocessing/preprocess.py`

- Vérifie les valeurs manquantes (aucune pour ce dataset)
- Vérifie les doublons
- Valide que les données sont correctes

```python
from src.preprocessing.preprocess import preprocess_data

df_clean = preprocess_data(df)
```

### Étape 3 : Feature Engineering
**Fichier** : `src/features/build_features.py`

- Sélectionne les 6 features importantes :
  - `is_foreign_transaction`
  - `account_age_days`
  - `country_risk`
  - `num_transactions_24h`
  - `transaction_amount`
  - `device_type`

- Encode les variables catégorielles :
  - `country_risk` → One-Hot Encoding (3 colonnes)
  - `device_type` → One-Hot Encoding (3 colonnes)

- Sépare X (features) et y (cible)

```python
from src.features.build_features import build_features

X, y = build_features(df_clean)
```

### Étape 4 : Entraînement
**Fichier** : `src/models/train.py`

1. **Split Train/Test (80/20)**
   - Stratifié pour garder la même proportion de fraudes

2. **Optimisation Optuna (30 trials)**
   - Optimise les hyperparamètres de Random Forest
   - Maximise le F1-Score
   - Teste différentes combinaisons

3. **Entraînement du meilleur modèle**
   - Utilise les meilleurs paramètres trouvés
   - `class_weight='balanced'` pour gérer le déséquilibre

4. **Évaluation sur le test set**
   - Accuracy, Precision, Recall, F1, ROC-AUC
   - Matrice de confusion

5. **Tracking MLflow**
   - Enregistre tous les paramètres
   - Enregistre toutes les métriques
   - Sauvegarde le modèle

6. **Sauvegarde**
   - Modèle : `artifacts/models/random_forest_model.pkl`

```python
from src.models.train import train_with_mlflow

model, metrics = train_with_mlflow(X, y, n_trials=30)
```

## 📈 Visualiser les résultats avec MLflow

### Lancer l'interface MLflow

```bash
mlflow ui
```

Puis ouvrir : http://localhost:5000

### Ce que vous verrez

- **Experiments** : Toutes vos expériences
- **Runs** : Chaque exécution du pipeline
- **Parameters** : Hyperparamètres testés
- **Metrics** : Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts** : Modèle sauvegardé

## 🎯 Résultats Attendus

Après l'exécution, vous obtiendrez :

### 1. Modèle entraîné
```
artifacts/models/random_forest_model.pkl
```

### 2. Métriques affichées
```
📊 Résultats sur le Test Set :
  Accuracy  : 0.9XXX
  Precision : 0.XXX
  Recall    : 0.XXX ⭐
  F1-Score  : 0.XXX
  ROC-AUC   : 0.XXX

  Confusion Matrix :
    TN=XXXX | FP=XX
    FN=XX   | TP=XX
```

### 3. Tracking MLflow
- Expérience créée dans `mlruns/`
- Tous les paramètres et métriques enregistrés
- Modèle versionné et tracké

## 🔧 Tester les modules individuellement

### Tester le chargement
```bash
python -m src.data.load_data
```

### Tester le prétraitement
```bash
python -m src.preprocessing.preprocess
```

### Tester le feature engineering
```bash
python -m src.features.build_features
```

### Tester l'entraînement
```bash
python -m src.models.train
```

## 💡 Conseils

### Pour un test rapide
```bash
python run_pipeline.py --trials 10
```
- Seulement 10 essais Optuna
- Exécution en ~2-3 minutes

### Pour de meilleures performances
```bash
python run_pipeline.py --trials 100
```
- 100 essais Optuna
- Exécution en ~15-20 minutes
- Meilleurs hyperparamètres

### Pour plusieurs expériences
```bash
python run_pipeline.py --experiment "Experiment 1" --trials 30
python run_pipeline.py --experiment "Experiment 2" --trials 50
python run_pipeline.py --experiment "Experiment 3" --trials 100
```
- Toutes les expériences sont trackées dans MLflow
- Facile de comparer les résultats

## 📝 Configuration

Modifiez `configs/config.yaml` pour changer :
- Les features sélectionnées
- Les plages de recherche Optuna
- Les paramètres de split
- Les chemins de sauvegarde

## ✅ Avantages de cette architecture

1. **Modulaire** : Chaque étape est dans un fichier séparé
2. **Testable** : Chaque module peut être testé individuellement
3. **Réutilisable** : Les fonctions peuvent être importées ailleurs
4. **Clair** : Code simple avec commentaires explicatifs
5. **Tracké** : MLflow enregistre tout automatiquement
6. **Reproductible** : Même random_state = mêmes résultats

## 🐛 Troubleshooting

### Erreur : "File not found"
→ Vérifiez le chemin du fichier CSV

### Erreur : "Module not found"
→ Exécutez depuis la racine du projet

### MLflow UI ne démarre pas
→ Vérifiez que le port 5000 est libre

### Optuna trop lent
→ Réduisez `--trials` pour tester

## 🎓 Prochaines étapes

1. Créer une API FastAPI pour servir le modèle
2. Ajouter une interface Gradio
3. Déployer avec Docker
4. Ajouter des tests unitaires
5. Créer un CI/CD avec GitHub Actions
