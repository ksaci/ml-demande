# Modification : 10 Clusters par Défaut

## 🎯 Changement effectué

L'étape de **recherche du nombre optimal de clusters** a été **désactivée par défaut** pour améliorer les performances.

### Avant
```
📊 Recherche du nombre optimal de clusters...
  Test K=2... Inertie=1234, Silhouette=0.456
  Test K=3... Inertie=987, Silhouette=0.512
  ...
  Test K=10... Inertie=234, Silhouette=0.489
✓ K optimal suggéré : 8
```
⏱️ Durée : **~2-5 minutes par hôtel**

### Maintenant
```
💡 Configuration du clustering
  - Nombre de clusters : 10 (configuré)
  - Recherche automatique : DÉSACTIVÉE
```
⏱️ Durée : **Instantané** ✨

## 📊 Configuration actuelle

| Paramètre | Valeur |
|-----------|--------|
| Nombre de clusters | **10** (fixe) |
| Recherche automatique | **Désactivée** |
| Métrique clustering | DTW |

## ⚙️ Comment modifier le nombre de clusters

### Option 1 : Changer le nombre de clusters (recommandé)

Éditez `prediction_cluster.py` (ou les autres scripts) :

```python
# Options de clustering
N_CLUSTERS = 8  # ← Changez ici (ex: 5, 8, 12, 15...)
AUTO_FIND_K = False
```

### Option 2 : Activer la recherche automatique

```python
# Options de clustering
N_CLUSTERS = 10  # Ignoré si AUTO_FIND_K = True
AUTO_FIND_K = True  # ← Active la recherche automatique
```

⚠️ **Attention :** La recherche automatique prend **2-5 minutes** supplémentaires par hôtel.

## 📈 Impact sur les performances

### Analyse d'un seul hôtel
- **Avant :** ~10-15 minutes
- **Maintenant :** ~8-10 minutes
- **Gain :** 2-5 minutes ⚡

### Analyse en batch (4 hôtels)
- **Avant :** ~40-60 minutes
- **Maintenant :** ~32-40 minutes
- **Gain :** 8-20 minutes ⚡

## 🤔 Pourquoi 10 clusters ?

10 clusters est un **bon compromis** qui permet de :
- ✅ Capturer la diversité des profils de réservation
- ✅ Éviter le sur-clustering (trop de petits groupes)
- ✅ Rester interprétable pour l'analyse métier
- ✅ Fonctionner bien pour la plupart des hôtels

### Quand modifier le nombre de clusters ?

**Augmenter** (12, 15, 20) si :
- Vous avez beaucoup de données (>5000 courbes)
- Vous voulez une segmentation plus fine
- Les clusters actuels sont trop hétérogènes

**Diminuer** (5, 6, 8) si :
- Vous avez peu de données (<1000 courbes)
- Vous voulez une vue plus synthétique
- Les clusters actuels sont trop similaires

## 📝 Fichiers modifiés

| Fichier | Modification |
|---------|--------------|
| `prediction_cluster.py` | ✅ N_CLUSTERS=10, AUTO_FIND_K=False |
| `run_clustering_batch.py` | ✅ N_CLUSTERS=10, AUTO_FIND_K=False |
| `example_clustering_by_hotel.py` | ✅ N_CLUSTERS=10, AUTO_FIND_K=False |
| `README_CLUSTERING.md` | ✅ Documentation mise à jour |
| `CHANGELOG_CLUSTERING.md` | ✅ Version 2.1 ajoutée |

## 🔍 Vérification

Pour vérifier que la modification fonctionne :

```bash
python prediction_cluster.py D09
```

Vous devriez voir :
```
💡 ÉTAPE 5 : Configuration du clustering
  - Nombre de clusters : 10 (configuré)
  - Recherche automatique : DÉSACTIVÉE
  - Pour activer : AUTO_FIND_K = True
```

Et **PAS** :
```
💡 ÉTAPE 5 : Recherche du nombre optimal de clusters
  - Métrique : euclidean (rapide)
  ...
```

## 💡 Conseils

### Pour une analyse rapide
```python
N_CLUSTERS = 10
AUTO_FIND_K = False  # ← Recommandé pour le batch
```

### Pour une analyse optimale (plus lente)
```python
N_CLUSTERS = 10  # Valeur par défaut si recherche échoue
AUTO_FIND_K = True  # ← Recherche le meilleur K
```

### Pour tester différents K
```python
# Testez avec 5, 8, 10, 12 clusters
for k in [5, 8, 10, 12]:
    clustering = HotelBookingClustering(hotCode='D09')
    # ... analyse avec k clusters
```

## 📞 Questions fréquentes

**Q : Est-ce que 10 clusters est suffisant ?**  
R : Oui, pour la plupart des cas. Si vous avez un besoin spécifique, ajustez `N_CLUSTERS`.

**Q : Comment savoir si 10 est le bon nombre ?**  
R : Regardez les fichiers générés :
- `cluster_profiles.csv` : Distribution des courbes
- `clustering_comparison.png` : Si les profils sont trop similaires → réduire K
- `clustering_pca.png` : Si les clusters se chevauchent → ajuster K

**Q : Puis-je activer la recherche pour un seul hôtel ?**  
R : Oui, modifiez temporairement `AUTO_FIND_K = True` et relancez l'analyse.

**Q : Cela change-t-il la qualité du clustering ?**  
R : Non, seulement le nombre de clusters. La métrique DTW reste utilisée pour le clustering final.

---

**Date :** 16 novembre 2025  
**Version :** 2.1

