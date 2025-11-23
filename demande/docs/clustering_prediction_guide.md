# Guide d'utilisation du Clustering pour la Prédiction du Taux d'Occupation

## 📋 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Fonctionnalités ajoutées](#fonctionnalités-ajoutées)
3. [Fichiers générés](#fichiers-générés)
4. [Utilisation pour la prédiction](#utilisation-pour-la-prédiction)
5. [Exemples pratiques](#exemples-pratiques)
6. [Intégration dans un modèle ML](#intégration-dans-un-modèle-ml)

---

## 📊 Vue d'ensemble

Le script `prediction_cluster.py` effectue un clustering des courbes de montée en charge des réservations hôtelières (J-60 à J). Les nouvelles fonctionnalités permettent d'utiliser ces clusters pour prédire le taux d'occupation final d'une date de séjour, même avec des données incomplètes.

### Principe

1. **Entraînement** : Analyser les courbes historiques et identifier des profils types (clusters)
2. **Sauvegarde** : Enregistrer le modèle et les profils moyens de chaque cluster
3. **Prédiction** : Pour une nouvelle date avec une courbe incomplète :
   - Identifier le cluster le plus proche
   - Utiliser le profil moyen du cluster pour estimer le To final

---

## 🆕 Fonctionnalités ajoutées

### 1. `save_model(model_path='results/clustering_model.pkl')`

Sauvegarde le modèle de clustering (TimeSeriesKMeans) et le scaler.

**Contenu sauvegardé :**
- Modèle TimeSeriesKMeans entraîné
- Scaler (TimeSeriesScalerMeanVariance)
- Nombre optimal de clusters (K)
- Nombre de jours analysés (days_before)

**Exemple :**
```python
clustering.save_model('results/clustering_model.pkl')
```

---

### 2. `load_model(model_path='results/clustering_model.pkl')`

Charge un modèle de clustering précédemment sauvegardé.

**Exemple :**
```python
clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
clustering.load_model('results/clustering_model.pkl')
```

---

### 3. `save_cluster_profiles(output_path='results/cluster_profiles.csv')`

Sauvegarde les profils moyens de chaque cluster dans un fichier CSV.

**Structure du fichier CSV :**
- `cluster` : ID du cluster (0, 1, 2, ...)
- `n_samples` : Nombre de courbes dans ce cluster
- `percentage` : Pourcentage de courbes dans ce cluster
- `J-60_mean`, `J-59_mean`, ..., `J-0_mean` : Valeurs moyennes du To pour chaque jour
- `J-60_std`, `J-59_std`, ..., `J-0_std` : Écart-types pour chaque jour

**Exemple :**
```python
profiles_df = clustering.save_cluster_profiles()
```

---

### 4. `get_cluster_profile(cluster_id)`

Retourne le profil complet d'un cluster spécifique (moyenne, médiane, quartiles).

**Retour :**
```python
{
    'cluster': 0,
    'n_samples': 150,
    'mean_curve': {'J-60': 0.15, 'J-59': 0.16, ..., 'J-0': 0.85},
    'std_curve': {'J-60': 0.05, 'J-59': 0.05, ..., 'J-0': 0.10},
    'median_curve': {...},
    'q25_curve': {...},
    'q75_curve': {...}
}
```

**Exemple :**
```python
profile = clustering.get_cluster_profile(cluster_id=2)
print(f"To moyen à J : {profile['mean_curve']['J-0']}")
```

---

### 5. `predict_cluster(partial_curve, days_available=None)`

**La fonction principale pour la prédiction !**

Prédit le cluster d'une nouvelle date de séjour à partir d'une courbe incomplète.

**Paramètres :**
- `partial_curve` : Dictionnaire ou Series avec les valeurs de To disponibles
  - Format : `{'J-60': 0.1, 'J-59': 0.12, ..., 'J-15': 0.45}`
- `days_available` : Nombre de jours disponibles (optionnel, détecté automatiquement)

**Retour :**
```python
{
    'cluster': 2,                              # ID du cluster prédit
    'confidence': 0.85,                        # Score de confiance (0-1)
    'all_distances': {0: 2.3, 1: 1.5, 2: 0.8}, # Distances à tous les clusters
    'full_curve': {...}                        # Courbe complète (avec extrapolation)
}
```

**Algorithme :**
1. Compléter les valeurs manquantes par interpolation
2. Pour les valeurs futures (non observées) :
   - Calculer la distance à chaque cluster sur les jours disponibles
   - Identifier le cluster le plus proche
   - Extrapoler avec le profil moyen de ce cluster
3. Normaliser la courbe complète
4. Prédire le cluster avec le modèle TimeSeriesKMeans

**Exemple :**
```python
# On est à J-15 : on a les données de J-60 à J-15
partial_curve = {
    'J-60': 0.10, 'J-59': 0.11, ..., 'J-15': 0.45
}

prediction = clustering.predict_cluster(partial_curve)
print(f"Cluster prédit : {prediction['cluster']}")
print(f"Confiance : {prediction['confidence']:.3f}")
```

---

## 📁 Fichiers générés

Après l'exécution de `prediction_cluster.py`, les fichiers suivants sont générés dans `results/` :

| Fichier | Description | Utilisation |
|---------|-------------|-------------|
| `clustering_model.pkl` | Modèle de clustering + scaler | Charger avec `load_model()` |
| `cluster_profiles.csv` | Profils moyens des clusters | Features pour ML |
| `clustering_results.csv` | Toutes les courbes avec leur cluster | Données d'entraînement |
| `clustering_*.png` | Visualisations | Analyse exploratoire |

---

## 🔮 Utilisation pour la prédiction

### Scénario typique

**Problème :** Nous sommes à J-15 d'une date de séjour. On veut prédire le To final à J.

**Solution :**

```python
from prediction_cluster import HotelBookingClustering
import pandas as pd

# 1. Charger le modèle de clustering
clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
clustering.load_model('results/clustering_model.pkl')

# Charger les données pour avoir les profils
clustering.load_data()
clustering.prepare_booking_curves()

# 2. Préparer la courbe partielle (J-60 à J-15)
partial_curve = {
    'J-60': 0.12,
    'J-59': 0.13,
    # ... (valeurs de J-60 à J-15)
    'J-15': 0.45
}

# 3. Prédire le cluster
prediction = clustering.predict_cluster(partial_curve)
cluster_id = prediction['cluster']

print(f"Cluster prédit : {cluster_id}")
print(f"Confiance : {prediction['confidence']:.3f}")

# 4. Obtenir le profil moyen du cluster
profile = clustering.get_cluster_profile(cluster_id)

# 5. Estimer le To final
predicted_to_final = profile['mean_curve']['J-0']
predicted_to_std = profile['std_curve']['J-0']

print(f"\n📈 Prédiction du To final :")
print(f"  - To prédit : {predicted_to_final:.3f} ({predicted_to_final*100:.1f}%)")
print(f"  - Écart-type : {predicted_to_std:.3f}")
print(f"  - Intervalle [±1σ] : [{predicted_to_final - predicted_to_std:.3f}, {predicted_to_final + predicted_to_std:.3f}]")
```

---

## 💡 Exemples pratiques

Un script d'exemples complet est fourni : `example_predict_cluster.py`

### Exemple 1 : Prédiction avec données partielles

```python
from prediction_cluster import HotelBookingClustering

clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
clustering.load_model('results/clustering_model.pkl')
clustering.load_data()
clustering.prepare_booking_curves()

# Courbe incomplète (J-60 à J-30)
partial_curve = {'J-60': 0.15, 'J-59': 0.16, ..., 'J-30': 0.45}

prediction = clustering.predict_cluster(partial_curve)
profile = clustering.get_cluster_profile(prediction['cluster'])
```

### Exemple 2 : Charger les profils depuis CSV

```python
import pandas as pd

# Charger directement les profils (sans le modèle)
profiles_df = pd.read_csv('results/cluster_profiles.csv', sep=';')

# Obtenir le profil du cluster 0
cluster_0 = profiles_df[profiles_df['cluster'] == 0].iloc[0]
print(f"To moyen à J : {cluster_0['J-0_mean']:.3f}")
```

### Exemple 3 : Utiliser dans un modèle de prédiction

Voir la section suivante.

---

## 🤖 Intégration dans un modèle ML

### Approche 1 : Cluster comme feature catégorielle (One-Hot Encoding)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 1. Charger les résultats de clustering
results_df = pd.read_csv('results/clustering_results.csv', sep=';')

# 2. Créer les features
# Features temporelles
temporal_features = ['J-60', 'J-45', 'J-30', 'J-15', 'J-7']

# Features de croissance
results_df['growth_60_30'] = results_df['J-30'] - results_df['J-60']
results_df['growth_30_15'] = results_df['J-15'] - results_df['J-30']
results_df['growth_15_7'] = results_df['J-7'] - results_df['J-15']

# One-hot encoding du cluster
cluster_dummies = pd.get_dummies(results_df['cluster'], prefix='cluster')

# 3. Combiner toutes les features
X = pd.concat([
    results_df[temporal_features + ['growth_60_30', 'growth_30_15', 'growth_15_7']],
    cluster_dummies
], axis=1)

y = results_df['J-0']  # Target : To final

# 4. Entraîner le modèle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 5. Évaluer
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R² : {r2_score(y_test, y_pred):.4f}")
```

### Approche 2 : Utiliser les profils moyens comme features

```python
import pandas as pd

# Charger les résultats et les profils
results_df = pd.read_csv('results/clustering_results.csv', sep=';')
profiles_df = pd.read_csv('results/cluster_profiles.csv', sep=';')

# Fusionner avec les profils moyens
# Ajouter les profils moyens comme features pour chaque date
results_with_profiles = results_df.merge(
    profiles_df[['cluster', 'J-0_mean', 'J-0_std', 'percentage']],
    on='cluster',
    how='left'
)

# Créer les features
X = results_with_profiles[[
    'J-60', 'J-45', 'J-30', 'J-15', 'J-7',
    'J-0_mean',      # To moyen attendu du cluster
    'J-0_std',       # Volatilité du cluster
    'percentage'     # Importance du cluster
]]

y = results_with_profiles['J-0']

# Entraîner le modèle...
```

### Approche 3 : Utiliser la distance aux clusters

```python
# Pour chaque courbe, calculer la distance à chaque centre de cluster
# → Donne une mesure de "similarité" à chaque profil

from prediction_cluster import HotelBookingClustering

clustering = HotelBookingClustering(...)
clustering.load_model('results/clustering_model.pkl')
# ...

# Pour une nouvelle date
partial_curve = {...}
prediction = clustering.predict_cluster(partial_curve)

# Utiliser les distances comme features
distances = prediction['all_distances']
# distances = {0: 2.3, 1: 1.5, 2: 0.8, ...}

# Features : [J-60, ..., J-15, dist_cluster_0, dist_cluster_1, ..., dist_cluster_K]
```

---

## 🎯 Recommandations

### Pour l'entraînement du modèle ML

1. **Utiliser le cluster comme feature** : 
   - ✅ One-hot encoding du cluster
   - ✅ Profil moyen du cluster (J-0_mean, J-0_std)
   - ✅ Distance aux clusters

2. **Features temporelles importantes** :
   - To à différents moments : J-60, J-45, J-30, J-15, J-7
   - Croissances : growth_60_30, growth_30_15, growth_15_7
   - Pentes : (J-15 - J-30) / 15

3. **Features de contexte** :
   - Jour de la semaine de la date de séjour
   - Mois / saison
   - Jours fériés / événements
   - Code hôtel (one-hot ou embedding)

### Pour la prédiction en production

1. **Charger le modèle une seule fois** au démarrage
2. **Pour chaque nouvelle date** :
   - Récupérer la courbe partielle depuis la BDD
   - Prédire le cluster avec `predict_cluster()`
   - Ajouter le cluster comme feature
   - Prédire le To final avec votre modèle ML

3. **Mettre à jour le modèle de clustering régulièrement** (tous les 3-6 mois)

---

## 📞 Support

Pour toute question sur l'utilisation du clustering dans votre modèle de prédiction, consultez :
- `example_predict_cluster.py` : Exemples complets
- `prediction_cluster.py` : Code source avec documentation détaillée

---

**Auteur** : Script de clustering pour la prédiction du taux d'occupation  
**Date** : Novembre 2025  
**Version** : 1.0

