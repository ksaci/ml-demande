# 🕒 TimeSeriesKMeans avec DTW - Guide d'Utilisation

## 📊 Changement Important : KMeans → TimeSeriesKMeans

Le script a été mis à jour pour utiliser **TimeSeriesKMeans** de la bibliothèque `tslearn` au lieu de KMeans classique de scikit-learn.

## 🎯 Pourquoi TimeSeriesKMeans ?

### Problème avec KMeans classique

KMeans utilise la **distance euclidienne** qui n'est pas adaptée aux séries temporelles :

```python
# Distance euclidienne : somme des carrés des différences point par point
distance = sqrt(sum((point1[i] - point2[i])^2))
```

**Limitations** :
- ❌ Ne gère pas les décalages temporels
- ❌ Sensible aux variations d'échelle
- ❌ Ne capture pas les formes similaires

### Avantage de TimeSeriesKMeans avec DTW

**DTW (Dynamic Time Warping)** aligne les séries temporelles de manière optimale :

```python
# DTW trouve le meilleur alignement entre deux courbes
# Même si elles sont décalées ou de vitesses différentes
```

**Avantages** :
- ✅ Gère les décalages temporels
- ✅ Capture les formes similaires
- ✅ Plus robuste pour les séries temporelles
- ✅ Meilleure qualité de clustering

## 📦 Installation

```bash
pip install tslearn
```

## 🔧 Modifications Apportées

### 1. Imports

**Avant** :
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

**Maintenant** :
```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
```

### 2. Format des Données

TimeSeriesKMeans attend un format 3D : `(n_samples, n_timestamps, n_features)`

**Conversion automatique** :
```python
# De (n_courbes, n_jours) à (n_courbes, n_jours, 1)
data_3d = curves_df[feature_cols].values[:, :, np.newaxis]
```

### 3. Utilisation

#### Dans le script `prediction_cluster.py`

```python
# Clustering avec DTW (métrique par défaut)
clustering.perform_clustering(n_clusters=5, metric="dtw")

# Ou tester d'autres métriques :
clustering.perform_clustering(n_clusters=5, metric="euclidean")
clustering.perform_clustering(n_clusters=5, metric="softdtw")
```

#### Dans le notebook autonome

Tout le code est visible et modifiable :

```python
# Dans la cellule de clustering
ts_kmeans_model = TimeSeriesKMeans(
    n_clusters=5, 
    metric="dtw",      # DTW pour séries temporelles
    random_state=42, 
    n_init=10,
    verbose=False
)
```

## 📏 Métriques Disponibles

### 1. **DTW (Dynamic Time Warping)** - Recommandé
```python
metric="dtw"
```
- ✅ Meilleur pour les séries temporelles
- ✅ Gère les décalages
- ⚠️ Plus lent que euclidean

### 2. **Euclidean**
```python
metric="euclidean"
```
- ✅ Plus rapide
- ❌ Moins adapté aux séries temporelles

### 3. **Soft-DTW**
```python
metric="softdtw"
```
- ✅ Version différentiable de DTW
- ✅ Bon compromis vitesse/qualité

## 🎨 Exemple Visuel

### Avec KMeans classique (distance euclidienne)

```
Courbe A:  ___/‾‾‾‾\___
Courbe B:  __/‾‾‾‾\____  (légèrement décalée)
         
Distance euclidienne : GRANDE (points ne s'alignent pas)
→ Clusters différents ❌
```

### Avec TimeSeriesKMeans + DTW

```
Courbe A:  ___/‾‾‾‾\___
Courbe B:  __/‾‾‾‾\____
         
DTW aligne les courbes intelligemment
→ Même cluster ✅
```

## 📊 Impact sur les Résultats

Avec TimeSeriesKMeans + DTW, vous obtiendrez :

1. **Meilleurs clusters** : Courbes similaires regroupées ensemble
2. **Profils plus clairs** : "Dernière minute" vs "Anticipé" mieux séparés
3. **Métriques améliorées** : Score de silhouette généralement plus élevé

## ⚙️ Paramètres Ajustables

### Dans `prediction_cluster.py`

```python
# Changer la métrique
optimal_k = clustering.find_optimal_clusters(max_k=10, metric="dtw")
clustering.perform_clustering(n_clusters=optimal_k, metric="dtw")

# Métriques disponibles
metrics = ["dtw", "euclidean", "softdtw"]
```

### Dans le notebook

Modifiez directement dans les cellules :

```python
# Cellule de recherche K optimal
for k in k_range:
    ts_kmeans = TimeSeriesKMeans(
        n_clusters=k, 
        metric="dtw",          # ⭐ Changez ici
        random_state=42, 
        n_init=5,
        verbose=False
    )
```

## 🚀 Performance

### Temps d'exécution

| Métrique | Vitesse | Qualité |
|----------|---------|---------|
| euclidean | ⚡⚡⚡ Rapide | ⭐⭐ Moyenne |
| softdtw | ⚡⚡ Moyen | ⭐⭐⭐ Bonne |
| **dtw** | ⚡ Lent | ⭐⭐⭐⭐ **Excellente** |

**Recommandation** : Utilisez DTW pour l'analyse finale, euclidean pour les tests rapides.

## 📝 Notes Techniques

### Format des données

```python
# Input attendu par TimeSeriesKMeans
shape: (n_samples, n_timestamps, n_features)
exemple: (5000, 61, 1)
         ↑      ↑   ↑
         |      |   └─ 1 feature (le To)
         |      └───── 61 points (J-60 à J)
         └────────────── 5000 courbes
```

### Normalisation

`TimeSeriesScalerMeanVariance` normalise chaque série temporelle :
- Moyenne = 0
- Écart-type = 1

## 🔍 Débogage

Si vous rencontrez des erreurs :

### Erreur : "Module 'tslearn' not found"
```bash
pip install tslearn
```

### Erreur : "Shape mismatch"
Vérifiez que les données sont bien en 3D :
```python
print(scaled_curves.shape)  # Doit être (n, timestamps, 1)
```

### DTW trop lent ?
Réduisez les données ou utilisez `softdtw` :
```python
clustering.perform_clustering(n_clusters=5, metric="softdtw")
```

## 📚 Ressources

- [tslearn documentation](https://tslearn.readthedocs.io/)
- [DTW expliqué](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [TimeSeriesKMeans API](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html)

## ✅ Checklist

- [x] tslearn installé
- [x] Script mis à jour avec TimeSeriesKMeans
- [x] Notebook autonome mis à jour
- [x] DTW comme métrique par défaut
- [x] Format 3D pour les données
- [x] Tests effectués

---

**Version** : 2.0 (avec TimeSeriesKMeans)  
**Date** : Novembre 2025  
**Métrique recommandée** : DTW

