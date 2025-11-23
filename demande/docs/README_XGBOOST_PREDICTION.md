# 🏨 Prédiction du Taux d'Occupation Futur avec XGBoost

Ce script utilise les résultats du clustering pour entraîner un modèle XGBoost qui prédit le taux d'occupation (To) futur des hôtels.

## 🎯 Objectif

Prédire le taux d'occupation d'un hôtel dans le futur (J+7, J+14, etc.) en utilisant :
- Les courbes de montée en charge clusterisées
- Les données PM (Prix Moyen) actuelles
- Les données RevPAR (Revenue Per Available Room) actuelles
- L'appartenance aux clusters identifiés

## 📊 Données d'entrée

### 1. Résultats du clustering (`results/clustering_results.csv`)
- **hotCode** : Code de l'hôtel
- **stay_date** : Date de séjour
- **J-60 à J-0** : Valeurs du taux d'occupation pour chaque jour avant la date de séjour
- **cluster** : Numéro du cluster assigné

### 2. Données indicateurs (`data/*.csv`)
- **Pm** : Prix Moyen actuel
- **revpz** : RevPAR (Revenue Per Available Room) actuel

## 🚀 Utilisation

### Entraînement du modèle

```bash
cd c:\github\machineLearning\demande
python xgboost_to_prediction.py
```

### Configuration

Modifiez les paramètres dans la fonction `main()` :

```python
# Fichier des résultats de clustering
CLUSTERING_RESULTS = 'results/clustering_results.csv'

# Dossier contenant les fichiers indicateurs
INDICATEURS_DIR = 'data'

# Horizon de prédiction (en jours)
PREDICTION_HORIZON = 7  # Prédire To à J+7
```

## 🏗️ Architecture du modèle

### Features utilisées
1. **Courbe de To récente** : Valeurs J-7 à J-37 (30 jours)
2. **Cluster** : Appartenance au cluster identifié
3. **PM actuel** : Prix Moyen du jour de séjour
4. **RevPAR actuel** : Revenue Per Available Room du jour

### Cible (Target)
- **To futur** : Taux d'occupation prédit à J+`PREDICTION_HORIZON`

### Approches de préparation des données

#### 1. Approche principale
- Fusion directe des données clustering + PM/RevPAR
- Recherche des valeurs To futures dans les données historiques
- Calcul d'une approximation To = PM_futur / RevPAR_futur

#### 2. Approche alternative (si peu de données)
- Utilisation des patterns moyens par cluster
- Prédiction basée sur l'évolution typique du cluster

## 📈 Évaluation du modèle

### Métriques
- **MAE** (Mean Absolute Error) : Erreur absolue moyenne
- **RMSE** (Root Mean Square Error) : Racine de l'erreur quadratique moyenne
- **R² Score** : Coefficient de détermination

### Validation croisée
- Validation 5-fold pour évaluer la robustesse
- Comparaison train/test pour détecter le surapprentissage

### Analyse des features importantes
- Graphique des 20 features les plus importantes
- Sauvegardé automatiquement dans `results/feature_importance.png`

## 💾 Sauvegarde du modèle

Le modèle entraîné est sauvegardé dans :
```
models/xgboost_to_predictor.pkl
```

Contient :
- Le modèle XGBoost entraîné
- Le scaler pour la normalisation
- La liste des features utilisées

## 🔮 Utilisation du modèle entraîné

```python
from xgboost_to_prediction import ToPredictor

# Charger le modèle
predictor = ToPredictor.load_model('models/xgboost_to_predictor.pkl')

# Préparer les données de prédiction
curve_data = {'J-7': 0.85, 'J-6': 0.87, 'J-5': 0.89, ...}  # 30 valeurs
pm_current = 120.5
revpar_current = 95.2

# Prédire
future_to = predictor.predict_future_to(
    hotel_code='ABC',
    current_date=pd.Timestamp('2024-01-15'),
    curve_data=curve_data,
    pm_current=pm_current,
    revpar_current=revpar_current,
    prediction_horizon=7
)

print(f"To prédit à J+7 : {future_to:.3f}")
```

## ⚙️ Configuration XGBoost

```python
self.model = xgb.XGBRegressor(
    n_estimators=200,      # Nombre d'arbres
    max_depth=6,           # Profondeur maximale
    learning_rate=0.1,     # Taux d'apprentissage
    subsample=0.8,         # Fraction des échantillons
    colsample_bytree=0.8,  # Fraction des features
    random_state=42,
    n_jobs=-1             # Utilisation de tous les CPU
)
```

## 📁 Structure des fichiers générés

```
demande/
├── models/
│   └── xgboost_to_predictor.pkl     # Modèle sauvegardé
├── results/
│   └── feature_importance.png       # Importance des features
└── xgboost_to_prediction.py         # Script principal
```

## 🔧 Personnalisation

### Changer l'horizon de prédiction
```python
PREDICTION_HORIZON = 14  # Prédire à J+14 au lieu de J+7
```

### Modifier les hyperparamètres XGBoost
Ajustez les paramètres dans la méthode `train_model()` pour optimiser les performances.

### Ajouter des features
Modifiez `prepare_features()` pour inclure d'autres variables prédictives (météo, événements, saisonnalité, etc.).

## 🎯 Performance attendue

- **MAE typique** : 0.02 - 0.05 (2-5% d'erreur absolue)
- **R² typique** : 0.75 - 0.90
- **Temps d'entraînement** : 2-5 minutes selon la taille des données

## 🚨 Dépannage

### Erreur "Aucune donnée cible trouvée"
- Vérifiez que les dates dans `clustering_results.csv` correspondent aux dates dans `indicateurs.csv`
- Le script bascule automatiquement sur l'approche alternative

### Erreur "Peu d'échantillons"
- L'approche alternative est utilisée automatiquement
- Considérez réduire `PREDICTION_HORIZON` ou utiliser plus de données historiques

### Modèle qui overfit
- Réduisez `max_depth` ou `n_estimators`
- Augmentez `subsample` et `colsample_bytree`
