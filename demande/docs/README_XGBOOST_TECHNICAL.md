# 🏨 Documentation Technique - Prédiction XGBoost du Taux d'Occupation

## Vue d'ensemble

Ce document détaille l'architecture technique et le fonctionnement interne du script `xgboost_to_prediction.py`, qui implémente un système de prédiction du taux d'occupation (To) futur des hôtels utilisant l'algorithme XGBoost.

## Architecture du système

### 1. Structure des classes

#### Classe `ToPredictor`
Classe principale orchestrant tout le processus de prédiction.

**Attributs :**
- `clustering_results_path` : Chemin vers le fichier CSV des résultats de clustering
- `indicateurs_path` : Chemin vers le dossier contenant les fichiers indicateurs
- `model` : Modèle XGBoost entraîné (None si non entraîné)
- `scaler` : Objet StandardScaler pour la normalisation
- `feature_columns` : Liste des noms des colonnes de features utilisées

**Méthodes principales :**
- `load_data()` : Chargement et validation des données
- `prepare_features()` : Préparation des features d'entraînement
- `train_model()` : Entraînement du modèle XGBoost
- `predict_future_to()` : Prédiction pour de nouvelles données
- `plot_feature_importance()` : Analyse et visualisation des features importantes
- `save_model()` : Sauvegarde du modèle entraîné

## Algorithme de prédiction

### 1. Problème de prédiction

**Type :** Régression supervisée
**Entrée :** Données historiques d'un hôtel sur 60 jours + métriques actuelles
**Sortie :** Taux d'occupation prédit à J+N jours (N configurable, défaut 7)

### 2. Features utilisées

#### Features temporelles (Courbes To)
- `J-60` à `J-0` : Valeurs du taux d'occupation pour chaque jour des 60 derniers jours
- **Raison :** Capturer les patterns saisonniers et les tendances d'évolution

#### Features économiques (valeurs actuelles)
- `Pm_current` : Prix Moyen actuel (une seule valeur)
- `RevPAR_current` : Revenue Per Available Room actuel (une seule valeur)
- **Raison :** Indicateurs économiques de référence pour la prédiction

#### Features catégorielles
- `cluster` : Appartenance au cluster identifié par l'algorithme de clustering
- **Raison :** Différents types d'hôtels suivent des patterns différents

### 3. Variable cible (Target)
**Formule :** `To_futur = PM_futur / RevPAR_futur`
- Calculée pour la date J + `prediction_horizon`
- Plafonnée à 200% pour éviter les valeurs aberrantes
- Fallback sur To actuel si données futures indisponibles

## Pipeline de traitement des données

### Phase 1 : Chargement des données

```python
def load_data(self):
    # 1. Charger clustering_results.csv
    cluster_df = pd.read_csv(self.clustering_results_path, sep=';')

    # 2. Charger indicateurs.csv
    indicateurs_df = pd.read_csv(f"{self.indicateurs_path}/Indicateurs.csv", sep=';')

    # 3. Conversion des dates
    cluster_df['stay_date'] = pd.to_datetime(cluster_df['stay_date'])
    indicateurs_df['Date'] = pd.to_datetime(indicateurs_df['Date'])
    indicateurs_df['ObsDate'] = pd.to_datetime(indicateurs_df['ObsDate'])

    return cluster_df, indicateurs_df
```

### Phase 2 : Préparation des features

#### Stratégie principale
Pour chaque date de séjour dans les résultats de clustering :

1. **Fusion des données :**
   - Jointure entre résultats de clustering et données indicateurs
   - Utilisation des valeurs PM et RevPAR de la date de séjour

2. **Construction des features :**
   - Courbe To complète (J-60 à J-0)
   - Valeur unique PM actuelle
   - Valeur unique RevPAR actuelle
   - Cluster d'appartenance

3. **Calcul de la cible :**
   - Recherche des données PM/RevPAR pour J + `prediction_horizon`
   - Calcul : `target_to = PM_futur / RevPAR_futur`

#### Stratégie alternative (fallback)
Si moins de 100 échantillons valides :
- Utilisation des patterns moyens par cluster
- Prédiction basée sur l'évolution historique du cluster

### Phase 3 : Nettoyage et validation

```python
# Suppression des lignes avec cible manquante
features_df = features_df.dropna(subset=['target_to'])

# Remplacement des NaN par moyennes (features) ou suppression (cible)
for col in numeric_cols:
    if col != 'target_to':
        mean_val = features_df[col].mean()
        features_df[col] = features_df[col].fillna(mean_val)
```

## Algorithme XGBoost

### Configuration du modèle

```python
self.model = xgb.XGBRegressor(
    n_estimators=200,      # Nombre d'arbres dans la forêt
    max_depth=6,           # Profondeur maximale des arbres
    learning_rate=0.1,     # Taux d'apprentissage
    subsample=0.8,         # Fraction des échantillons utilisés par arbre
    colsample_bytree=0.8,  # Fraction des features utilisés par arbre
    random_state=42,       # Reproductibilité
    n_jobs=-1             # Utilisation de tous les CPU disponibles
)
```

### Fonction de perte
**Objective :** `reg:squarederror` (régression avec erreur quadratique)
**Raison :** Adapté pour prédire des valeurs continues positives

### Optimisation
- **Boosting :** Gradient Boosting itératif
- **Régularisation :** L1/L2 implicite via la structure des arbres
- **Early stopping :** Non utilisé (ensemble fixe de 200 arbres)

## Évaluation et métriques

### Métriques principales

#### Mean Absolute Error (MAE)
```python
mae = mean_absolute_error(y_true, y_pred)
```
- **Interprétation :** Erreur absolue moyenne en points de pourcentage
- **Exemple :** MAE = 0.05 signifie erreur moyenne de 5% sur le To

#### Root Mean Square Error (RMSE)
```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```
- **Interprétation :** Racine de l'erreur quadratique moyenne
- **Sensibilité :** Pénalise plus les grandes erreurs

#### Coefficient de détermination (R²)
```python
r2 = r2_score(y_true, y_pred)
```
- **Interprétation :** Pourcentage de variance expliqué par le modèle
- **Plage :** 0 à 1 (1 = modèle parfait)

### Validation croisée

```python
cv_scores = cross_val_score(
    self.model, X_train_scaled, y_train,
    cv=5, scoring='neg_mean_absolute_error'
)
```
- **Stratégie :** 5-fold cross-validation
- **Métrique :** Negative MAE (convention scikit-learn)
- **Interprétation :** Robustesse du modèle sur différents sous-ensembles

## Normalisation des features

### StandardScaler
```python
self.scaler = StandardScaler()
X_scaled = self.scaler.fit_transform(X)
```

**Formule :** `X_scaled = (X - mean) / std`

- **Avantages :**
  - Features sur la même échelle
  - Améliore la convergence de XGBoost
  - Préserve les relations entre features

- **Features non normalisées :**
  - `cluster` : Variable catégorielle (pas de normalisation)

## Gestion des données manquantes

### Stratégie principale
1. **Cible (target_to) :** Suppression des lignes avec NaN
2. **Features numériques :** Remplacement par la moyenne
3. **Features catégorielles :** Remplacement par 0 ou mode

### Justification
- La cible ne peut pas être estimée si elle est manquante
- Les features peuvent être imputées sans biaiser excessivement le modèle
- Préférer la suppression à l'imputation pour la cible (qualité > quantité)

## Sauvegarde et chargement du modèle

### Format de sauvegarde
```python
model_data = {
    'model': self.model,           # Modèle XGBoost entraîné
    'scaler': self.scaler,         # StandardScaler ajusté
    'feature_columns': self.feature_columns  # Liste des features
}
joblib.dump(model_data, 'models/xgboost_to_predictor.pkl')
```

### Chargement
```python
model_data = joblib.load('models/xgboost_to_predictor.pkl')
predictor.model = model_data['model']
predictor.scaler = model_data['scaler']
predictor.feature_columns = model_data['feature_columns']
```

## Complexité algorithmique

### Entraînement
- **Temps :** O(n_estimators × n_samples × max_depth × n_features)
- **Mémoire :** O(n_samples × n_features + n_estimators × n_nodes)

### Prédiction
- **Temps :** O(n_estimators × max_depth)
- **Très rapide** une fois le modèle entraîné

### Optimisations
- `n_jobs=-1` : Utilisation de tous les CPU disponibles
- `subsample=0.8` : Réduction de la taille des échantillons par arbre
- `colsample_bytree=0.8` : Réduction du nombre de features par arbre

## Limitations et améliorations possibles

### Limitations actuelles
1. **Données futures :** Approximation To = PM/RevPAR (pas de To réel)
2. **Horizon fixe :** Un modèle par horizon de prédiction
3. **Features économiques :** Utilisation de valeurs uniques PM/RevPAR (pas de séries temporelles)
4. **Features limitées :** Pas d'intégration de données externes (météo, événements)
5. **Cluster statique :** Utilise le cluster déterminé historiquement

### Améliorations envisagées
1. **Multi-horizon :** Un seul modèle pour tous les horizons
2. **Features économiques :** Ajout de séries temporelles PM/RevPAR (au lieu de valeurs uniques)
3. **Features externes :** Intégration de données météo, calendriers, événements
4. **Cluster dynamique :** Classification automatique pour de nouveaux hôtels
5. **Probabiliste :** Prédiction d'intervalles de confiance
6. **Séries temporelles :** Utilisation d'algorithmes spécialisés (LSTM, Prophet)

## Débogage et monitoring

### Logs détaillés
- Progression du chargement des données
- Nombre d'échantillons à chaque étape
- Métriques d'évaluation complètes
- Erreurs avec traceback complet

### Points de contrôle
- Vérification de l'existence des fichiers
- Validation des types de données
- Contrôle de la qualité des features
- Tests de cohérence des prédictions

## Utilisation en production

### Prérequis
- Python 3.7+
- pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
- Fichiers de données : `clustering_results.csv`, `Indicateurs.csv`

### Déploiement
```bash
# Entraînement
python xgboost_to_prediction.py

# Utilisation du modèle entraîné
from xgboost_to_prediction import ToPredictor
predictor = ToPredictor.load_model('models/xgboost_to_predictor.pkl')
prediction = predictor.predict_future_to(hotel_code, current_date, curve_data, pm_current, revpar_current)
```

### Monitoring recommandé
- Validation périodique des performances
- Réentraînement avec nouvelles données
- Surveillance des distributions de prédictions
- Alertes sur dérive de données (data drift)

## Références techniques

### XGBoost
- [Documentation officielle](https://xgboost.readthedocs.io/)
- [Guide des hyperparamètres](https://xgboost.readthedocs.io/en/latest/parameter.html)

### Scikit-learn
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

### Pandas
- [Time series handling](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [DataFrame operations](https://pandas.pydata.org/docs/reference/frame.html)

---

*Document technique - Version 1.1 - Révisé avec valeurs uniques PM/RevPAR - Date : Décembre 2024*
