# Documentation Technique - Entraînement Modèle XGBoost

## 📋 Vue d'ensemble

Ce document décrit le système d'entraînement du modèle XGBoost pour la prédiction du taux d'occupation (TO) à J+7.

**Version:** 1.0  
**Date:** Décembre 2024  
**Auteur:** Équipe Data Science

---

## 🎯 Objectif

Prédire le taux d'occupation final (TO à J+0) en utilisant :
- Les courbes de montée en charge récentes (J-60 à J-7)
- Le numéro de cluster assigné
- Les features compressées des prix moyens (PM)
- Des features temporelles (mois, jour de la semaine)

---

## 🏗️ Architecture du Code

### Structure des fichiers

```
demande/
├── predictTo/                       # Dossier du projet PredictTO
│   ├── predictTo_train_model.py    # Script principal d'entraînement
│   ├── predictTo_predict_example.py # Exemple d'utilisation
│   ├── test_predictTo_setup.py     # Validation environnement
│   ├── load_predictTo_from_azure.py # Gestion Azure
│   ├── config_predictTo.yaml       # Configuration
│   ├── requirements_predictTo.txt  # Dépendances
│   ├── PREDICTTO_TRAINING_DOC.md   # Documentation technique
│   ├── README.md                   # Guide principal
│   └── predictTo_training.log      # Logs d'exécution
├── data/
│   └── Indicateurs.csv             # Données PM/RevPAR
└── results/
    ├── clustering_results.csv      # Résultats du clustering
    ├── models/                     # Modèles sauvegardés
    ├── xgb_scatter_plot.png       # Visualisations
    └── xgb_feature_importance.png
```

---

## 📊 Pipeline de Données

### 1. Chargement des Données

**Sources:**
- `results/clustering_results.csv` : Résultats du clustering avec colonnes TO (J-60 à J-0)
- `data/Indicateurs.csv` : Données des indicateurs PM et RevPAR

**Format attendu pour clustering_results.csv:**
```csv
hotCode;stay_date;nb_observations;J-60;J-59;...;J-0;cluster
D09;2022-01-01;61;0.213077;0.218891;...;0.665601;4
```

**Format attendu pour Indicateurs.csv:**
```csv
hotCode;Date;ObsDate;Pm;RevPAR
D09;2022-01-01;2021-11-02;148.48;95.2
```

### 2. Préparation des Données

#### Étape 2.1 : Calcul de la distance temporelle
```python
indicateurs["days_before"] = (indicateurs["Date"] - indicateurs["ObsDate"]).dt.days
```

#### Étape 2.2 : Pivot des PM
Transformation des observations PM en colonnes `pm_J-0`, `pm_J-1`, ..., `pm_J-60`

#### Étape 2.3 : Calcul des features PM compressées
À partir de la série temporelle PM (J-60 → J-8), on calcule :

| Feature | Description | Formule |
|---------|-------------|---------|
| `pm_mean` | Prix moyen | `mean(PM_series)` |
| `pm_slope` | Pente de la tendance | Régression linéaire |
| `pm_volatility` | Volatilité | `std(PM_series)` |
| `pm_diff_sum` | Somme des variations | `sum(abs(diff(PM_series)))` |
| `pm_change_ratio` | Ratio de changement | `(PM_last - PM_first) / PM_first` |
| `pm_last_jump` | Variation récente | `PM_last - PM[-6]` |
| `pm_trend_changes` | Nb changements de direction | Comptage des inversions de tendance |

#### Étape 2.4 : Features temporelles
```python
df["month"] = df["stay_date"].dt.month          # Mois (1-12)
df["dayofweek"] = df["stay_date"].dt.dayofweek  # Jour semaine (0-6)
```

### 3. Construction des Features

**Liste complète des features:**

1. **TO historiques (53 features):** `J-60`, `J-59`, ..., `J-8`
2. **PM compressées (7 features):** 
   - `pm_mean`, `pm_slope`, `pm_volatility`, `pm_diff_sum`
   - `pm_change_ratio`, `pm_last_jump`, `pm_trend_changes`
3. **Features additionnelles (4 features):**
   - `nb_observations` : Nombre d'observations
   - `cluster` : Numéro de cluster (0-N)
   - `month` : Mois du séjour
   - `dayofweek` : Jour de la semaine

**Total : 64 features**

**Variable cible:** `J-0` (Taux d'occupation final)

---

## 🤖 Modèle XGBoost

### Configuration par défaut

```python
{
    'n_estimators': 600,        # Nombre d'arbres
    'learning_rate': 0.05,      # Taux d'apprentissage
    'max_depth': 7,             # Profondeur max des arbres
    'subsample': 0.9,           # Échantillonnage des lignes
    'colsample_bytree': 0.9,    # Échantillonnage des colonnes
    'min_child_weight': 1,      # Poids minimum des feuilles
    'reg_lambda': 1.0,          # Régularisation L2
    'n_jobs': -1,               # Utiliser tous les CPU
    'random_state': 42          # Reproductibilité
}
```

### Prétraitement

**Normalisation StandardScaler:**
```python
X_scaled = StandardScaler().fit_transform(X)
```
- Moyenne = 0
- Écart-type = 1
- Appliqué sur toutes les features

### Split Train/Test

- **Train:** 80% des données
- **Test:** 20% des données
- **Méthode:** Split aléatoire stratifié (`random_state=42`)

---

## 📈 Métriques d'Évaluation

### Métriques calculées

1. **MAE (Mean Absolute Error)**
   ```
   MAE = mean(|y_true - y_pred|)
   ```
   - Erreur moyenne en points de TO
   - Exemple : MAE = 0.056 → erreur moyenne de 5.6%

2. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = sqrt(mean((y_true - y_pred)²))
   ```
   - Pénalise plus fortement les grandes erreurs

3. **R² (Coefficient de détermination)**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   - Proportion de variance expliquée
   - R² = 0.83 → 83% de la variance expliquée

### Résultats attendus

| Métrique | Train | Test |
|----------|-------|------|
| MAE | ~0.045 | ~0.056 |
| RMSE | ~0.062 | ~0.075 |
| R² | ~0.89 | ~0.83 |

### Feature Importance

Les features les plus importantes sont généralement :
1. `J-8` (TO à J-8) : ~46%
2. `J-9` (TO à J-9) : ~13%
3. `cluster` : ~8%
4. `J-23` : ~3%
5. `pm_change_ratio` : ~1%

---

## ☁️ Sauvegarde Azure Blob Storage

### Configuration requise

**Variable d'environnement:**
```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
```

### Structure de sauvegarde

```
Container: prediction-demande
└── models/
    └── 20241216_143025/          # Timestamp de l'entraînement
        ├── xgb_to_predictor.joblib
        ├── xgb_scaler.joblib
        └── feature_columns.txt
```

### Fichiers sauvegardés

1. **xgb_to_predictor.joblib**
   - Modèle XGBoost entraîné
   - Format : joblib (pickle optimisé)
   - Taille : ~2-5 MB

2. **xgb_scaler.joblib**
   - StandardScaler ajusté
   - Contient les paramètres de normalisation

3. **feature_columns.txt**
   - Liste ordonnée des features
   - Crucial pour la prédiction

---

## 🚀 Utilisation

### Installation des dépendances

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn azure-storage-blob
```

### Exécution du script

```bash
# Méthode 1 : Exécution directe
python demande/xgboost_train_model.py

# Méthode 2 : Avec variables d'environnement
export AZURE_STORAGE_CONNECTION_STRING="..."
python demande/xgboost_train_model.py
```

### Utilisation programmatique

```python
from xgboost_train_model import XGBoostOccupancyPredictor

# Configuration
config = {
    'clustering_results_path': 'results/clustering_results.csv',
    'indicateurs_path': 'data/Indicateurs.csv',
    'prediction_horizon': 7,
    'test_size': 0.2,
    'random_state': 42
}

# Instanciation
predictor = XGBoostOccupancyPredictor(config)

# Pipeline complet
clusters, indicateurs = predictor.load_data()
df = predictor.prepare_data(clusters, indicateurs)
X, y = predictor.create_features_target(df)
results = predictor.train_model(X, y)
predictor.evaluate_model(save_plots=True)
predictor.save_model_locally()
```

---

## 🔍 Monitoring et Logs

### Fichier de logs

**Emplacement:** `xgboost_training.log`

**Format:**
```
2024-12-16 14:30:25 - __main__ - INFO - Initialisation du XGBoostOccupancyPredictor
2024-12-16 14:30:26 - __main__ - INFO - Chargement des données...
2024-12-16 14:30:28 - __main__ - INFO - Clusters chargés: (1415, 65)
```

### Informations loggées

- ✅ Étapes du pipeline
- 📊 Métriques d'entraînement
- ⚠️ Warnings (données manquantes, etc.)
- ❌ Erreurs avec stack trace complet

---

## 🛠️ Maintenance et Évolution

### Réentraînement du modèle

**Fréquence recommandée:** Mensuelle ou lorsque :
- Nouvelles données disponibles (> 10% volume)
- Performance dégradée (MAE > seuil)
- Changement de saisonnalité

### Ajout de nouvelles features

1. Modifier `create_features_target()` pour ajouter les features
2. Mettre à jour `feature_cols`
3. Ré-entraîner le modèle
4. Comparer les performances

### Optimisation des hyperparamètres

Utiliser `GridSearchCV` ou `RandomizedSearchCV` :

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [400, 600, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
```

---

## 📚 Références

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Azure Blob Storage Python SDK](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python)

---

## ⚠️ Points d'attention

### Gestion des valeurs manquantes

- Les NaN dans les séries PM sont gérés par `compute_pm_features()`
- Les lignes avec NaN dans les features sont supprimées avant l'entraînement

### Reproductibilité

- Utiliser toujours le même `random_state`
- Vérifier que les données sources sont identiques
- Conserver les versions des bibliothèques

### Performance

- Temps d'entraînement : ~30-60 secondes (selon CPU)
- Taille du modèle : ~2-5 MB
- Temps de prédiction : <1ms par observation

---

## 🐛 Résolution de problèmes

### Erreur : "AZURE_STORAGE_CONNECTION_STRING non définie"

**Solution:** Définir la variable d'environnement ou ignorer la sauvegarde Azure

```bash
export AZURE_STORAGE_CONNECTION_STRING="votre_connection_string"
```

### Erreur : "La colonne 'J-0' est absente"

**Cause:** Fichier `clustering_results.csv` mal formaté  
**Solution:** Vérifier le format du CSV (séparateur `;`, colonnes TO présentes)

### Performance dégradée (R² < 0.7)

**Causes possibles:**
- Données insuffisantes
- Distribution différente (concept drift)
- Features manquantes

**Actions:**
- Augmenter le nombre de données
- Vérifier la qualité des clusters
- Ajouter des features pertinentes

---

## 📞 Contact

Pour toute question technique, contacter l'équipe Data Science.

---

**Dernière mise à jour:** 16 Décembre 2024

