# Changelog - Clustering par Hôtel

## Version 2.1.1 - Mode Développement Rapide (16 novembre 2025)

### 🚀 Changement de métrique pour le développement

#### Métrique EUCLIDEAN par défaut (au lieu de DTW)

**Raison :** DTW est trop lent pour le développement/testing rapide.

**Gain de performance :** ~2-3x plus rapide (5-8 min vs 15-25 min par hôtel)

**Nouvelle variable de configuration :**
```python
USE_DTW = False  # False = Euclidean (rapide) | True = DTW (qualité)
```

**Impact :**
- ⚡ **Développement :** EUCLIDEAN (rapide, USE_DTW=False)
- 🎯 **Production :** DTW (meilleure qualité, USE_DTW=True)

#### Fichiers modifiés

| Fichier | Configuration |
|---------|--------------|
| `prediction_cluster.py` | USE_DTW = False (Euclidean par défaut) |
| `run_clustering_batch.py` | USE_DTW = False (Euclidean par défaut) |
| `example_clustering_by_hotel.py` | USE_DTW = False (Euclidean par défaut) |
| `README_CLUSTERING.md` | Documentation mise à jour |
| `MODE_DEVELOPPEMENT.md` | **Nouveau** - Guide dev/prod |

#### Messages d'affichage

Avec `USE_DTW = False` (mode développement) :
```
💡 ÉTAPE 6 : Clustering final
  - Nombre de clusters : 10
  - Métrique : EUCLIDEAN (rapide - mode développement)
  - Initialisations : 10
  ⚠️  Mode développement - Changez USE_DTW = True pour la production
```

Avec `USE_DTW = True` (mode production) :
```
💡 ÉTAPE 6 : Clustering final
  - Nombre de clusters : 10
  - Métrique : DTW (meilleure qualité)
  - Initialisations : 5
```

#### Workflow recommandé

1. **Développement/Testing** : `USE_DTW = False` (rapide)
2. **Validation** : `AUTO_FIND_K = True` + `USE_DTW = False` (trouver K optimal)
3. **Production** : `USE_DTW = True` (meilleure qualité)

## Version 2.1 - Optimisation Performance (16 novembre 2025)

### 🚀 Changements de performance

#### Recherche du nombre optimal de clusters DÉSACTIVÉE par défaut

**Raison :** La recherche automatique du nombre optimal de clusters peut être lente, surtout avec DTW.

**Avant (v2.0) :**
```python
# Recherche automatique avec euclidean
optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
# Puis clustering avec DTW
```

**Maintenant (v2.1) :**
```python
# Nombre de clusters fixe (plus rapide)
N_CLUSTERS = 10  # Par défaut
AUTO_FIND_K = False  # Recherche désactivée

# Pour activer la recherche automatique :
AUTO_FIND_K = True
```

#### Nouvelles variables de configuration

Ajout de deux variables dans tous les scripts :

```python
# Options de clustering
N_CLUSTERS = 10  # Nombre de clusters (par défaut : 10)
AUTO_FIND_K = False  # Recherche automatique du nombre optimal (True pour activer)
```

**Fichiers modifiés :**
- `prediction_cluster.py` - Fonction `main()`
- `run_clustering_batch.py` - Configuration globale
- `example_clustering_by_hotel.py` - Configuration

#### Gains de performance

- ⚡ **~2-5 minutes économisées** par hôtel (pas de recherche K optimal)
- ⚡ **Meilleur pour le batch** : traiter plusieurs hôtels plus rapidement
- 🎯 **10 clusters** est un bon compromis pour la plupart des cas

#### Comment activer la recherche automatique

Si vous souhaitez laisser l'algorithme trouver le meilleur K :

```python
# Dans prediction_cluster.py, run_clustering_batch.py ou example_clustering_by_hotel.py
AUTO_FIND_K = True  # Activer la recherche automatique
```

### 📝 Détails techniques

**Messages d'affichage mis à jour :**

Avec `AUTO_FIND_K = False` :
```
💡 ÉTAPE 5 : Configuration du clustering
  - Nombre de clusters : 10 (configuré)
  - Recherche automatique : DÉSACTIVÉE
  - Pour activer : AUTO_FIND_K = True
```

Avec `AUTO_FIND_K = True` :
```
💡 ÉTAPE 5 : Recherche du nombre optimal de clusters
  - Métrique : euclidean (rapide)
  - Plage : K=2 à K=10
✓ K optimal suggéré : 8
```

**Fichiers générés :**

Le fichier `clustering_optimal_k.png` n'est généré que si `AUTO_FIND_K = True`.

## Version 2.0 - Analyse par Hôtel

### 🎯 Changements majeurs

#### 1. Code hôtel obligatoire en argument

**Avant :**
```bash
python prediction_cluster.py
# Demandait interactivement le chemin du fichier
```

**Maintenant :**
```bash
python prediction_cluster.py D09
# Code hôtel obligatoire en argument
```

#### 2. Structure des fichiers automatique

**Avant :**
- Fichier unique : `data/Indicateurs.csv`
- Résultats : `results/`

**Maintenant :**
- Par hôtel : `data/{hotCode}/Indicateurs.csv`
- Résultats : `results/{hotCode}/`

#### 3. Suppression des interactions utilisateur

**Avant :**
- Demandait la métrique pour trouver K optimal
- Demandait le nombre de clusters à utiliser
- Demandait la métrique pour le clustering final

**Maintenant :**
- Utilise automatiquement `euclidean` pour trouver K optimal
- Utilise le K optimal suggéré automatiquement
- Utilise automatiquement `DTW` pour le clustering final
- **Aucune interaction requise** - le script s'exécute en mode batch

### ✨ Nouvelles fonctionnalités

#### 1. Classe `HotelBookingClustering` améliorée

```python
# Nouvelle initialisation avec hotCode
clustering = HotelBookingClustering(hotCode='D09', days_before=60)

# L'ancienne méthode fonctionne toujours
clustering = HotelBookingClustering(csv_path='data/custom.csv', days_before=60)
```

#### 2. Scripts batch

Deux nouveaux scripts pour analyser plusieurs hôtels :

**PowerShell (Windows) :**
```powershell
.\run_clustering_batch.ps1
```

**Python (multiplateforme) :**
```bash
python run_clustering_batch.py
```

Modifiez la liste `HOTELS` dans ces scripts pour définir les hôtels à analyser.

#### 3. Documentation enrichie

- `README_CLUSTERING.md` - Guide rapide d'utilisation
- `docs/clustering_par_hotel.md` - Documentation complète
- `CHANGELOG_CLUSTERING.md` - Ce fichier

### 📝 Modifications détaillées

#### Fichier `prediction_cluster.py`

**Constructeur `__init__` :**
- Nouveau paramètre `hotCode` (optionnel)
- Chemin CSV automatique si `hotCode` est fourni
- Création automatique du dossier `results/{hotCode}/`
- Rétrocompatibilité avec `csv_path`

**Fonction `main()` :**
- Lit le code hôtel depuis `sys.argv[1]`
- Affiche un message d'erreur clair si manquant
- Validation du code hôtel (3 caractères)
- Utilise automatiquement la configuration optimale (euclidean → DTW)

**Méthodes de sauvegarde :**
- Toutes mises à jour pour utiliser `self.results_dir`
- Chemins par défaut : `results/{hotCode}/...`
- Paramètre `output_path` toujours optionnel

**Méthode `load_data()` :**
- Affiche le chemin du fichier chargé
- Affiche le code hôtel si disponible

**Méthode `save_model()` :**
- Sauvegarde également le `hotCode` dans le modèle

**Méthode `load_model()` :**
- Charge le `hotCode` si disponible (rétrocompatibilité)

### 🔄 Migration depuis la version précédente

#### Si vous utilisiez le script directement :

**Avant :**
```bash
python prediction_cluster.py
# Entrez le chemin : data/Indicateurs.csv
# Choisissez la métrique : 1
# Utilisez K=5 ? O
# Métrique finale : 1
```

**Maintenant :**
```bash
python prediction_cluster.py D09
# Aucune interaction - s'exécute automatiquement
```

#### Si vous utilisiez l'API Python :

**Avant :**
```python
clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
```

**Maintenant (recommandé) :**
```python
clustering = HotelBookingClustering(hotCode='D09', days_before=60)
```

**Ou (ancienne méthode toujours supportée) :**
```python
clustering = HotelBookingClustering(csv_path='data/custom.csv', days_before=60)
```

### 🛠️ Nouveaux fichiers

| Fichier | Description |
|---------|-------------|
| `run_clustering_batch.py` | Script Python pour analyse en batch |
| `run_clustering_batch.ps1` | Script PowerShell pour analyse en batch |
| `README_CLUSTERING.md` | Guide rapide d'utilisation |
| `docs/clustering_par_hotel.md` | Documentation complète |
| `CHANGELOG_CLUSTERING.md` | Ce fichier |

### ⚙️ Configuration par défaut

| Paramètre | Valeur |
|-----------|--------|
| `DAYS_BEFORE` | 60 (J-60 à J) |
| `YEAR_FILTER` | None (toutes les années) |
| `ENABLE_SMOOTHING` | True |
| `SMOOTHING_WINDOW` | 15 |
| `SMOOTHING_POLYORDER` | 3 |
| Métrique recherche K | `euclidean` |
| Métrique clustering | `DTW` |
| N_init (DTW) | 5 |

Ces paramètres peuvent être modifiés dans la fonction `main()` du script `prediction_cluster.py`.

### 🐛 Corrections de bugs

- Ajout de la gestion d'erreur si le fichier n'existe pas
- Validation du code hôtel (longueur)
- Messages d'erreur plus clairs
- Gestion de la rétrocompatibilité pour les anciens modèles

### 📊 Exemple d'utilisation complète

```bash
# 1. Analyser un hôtel
python prediction_cluster.py D09

# 2. Vérifier les résultats
ls results/D09/
# clustering_model.pkl
# clustering_results.csv
# cluster_profiles.csv
# *.png (graphiques)

# 3. Analyser plusieurs hôtels en batch
python run_clustering_batch.py

# 4. Utiliser le modèle sauvegardé
python
>>> from prediction_cluster import HotelBookingClustering
>>> clustering = HotelBookingClustering(hotCode='D09')
>>> clustering.load_model()
>>> result = clustering.predict_cluster({'J-60': 0.1, 'J-50': 0.2, 'J-30': 0.3})
>>> print(f"Cluster: {result['cluster']}, Confiance: {result['confidence']:.3f}")
```

### 🔮 Prochaines améliorations possibles

- [ ] Support des arguments en ligne de commande avancés (--days-before, --year-filter, etc.)
- [ ] Mode verbose/silencieux
- [ ] Export des résultats en JSON
- [ ] Interface web pour visualiser les résultats
- [ ] Comparaison automatique entre plusieurs hôtels
- [ ] Détection automatique des anomalies

### 📞 Support

Pour toute question ou problème :
1. Consultez `README_CLUSTERING.md` pour les exemples d'utilisation
2. Consultez `docs/clustering_par_hotel.md` pour la documentation complète
3. Vérifiez que la structure des fichiers est correcte
4. Vérifiez que le fichier `data/{hotCode}/Indicateurs.csv` existe

---

**Date de mise à jour :** 16 novembre 2025  
**Version :** 2.0.0

