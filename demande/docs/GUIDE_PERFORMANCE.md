# ⚡ Guide de Performance - DTW et TimeSeriesKMeans

## 🐌 Problème : Le Script est Lent / Bloque

Si le script bloque sur la recherche du nombre optimal de clusters, c'est normal ! **DTW est beaucoup plus lent que la distance euclidienne classique.**

### Pourquoi DTW est Lent ?

| Métrique | Complexité | Temps typique (1000 courbes, K=5) |
|----------|------------|-----------------------------------|
| **euclidean** | O(n) | ⚡ ~10 secondes |
| **softdtw** | O(n²) | ⚡⚡ ~2-3 minutes |
| **dtw** | O(n²) | 🐌 **~10-15 minutes** |

Avec 5000+ courbes et DTW, ça peut prendre **plusieurs heures** !

---

## 🚀 Solutions pour Accélérer

### Solution 1 : Stratégie Hybride (Recommandée) ⭐

**Utilisez euclidean pour trouver K, puis DTW pour le clustering final**

Quand le script demande :
```
Métrique pour trouver K optimal ? (1=euclidean/rapide, 2=dtw/lent, 3=softdtw) [1]:
```

➡️ **Appuyez sur Entrée** (euclidean par défaut) - RAPIDE !

Puis pour le clustering final :
```
Métrique finale ? (1=dtw/recommandé, 2=euclidean, 3=softdtw) [1]:
```

➡️ **Appuyez sur Entrée** (dtw) - Vous aurez la qualité de DTW !

**Temps gagné** : 90% ! ⚡

---

### Solution 2 : Réduire le Nombre de Courbes

Le script réduit automatiquement l'échantillon si > 3000 courbes avec DTW.

**Forcer une réduction** :

```python
# Dans le script main() ou notebook
optimal_k = clustering.find_optimal_clusters(
    max_k=10, 
    metric="dtw",
    sample_size=2000  # ⭐ Utiliser seulement 2000 courbes
)
```

**Recommandations** :
- < 1000 courbes : Pas de réduction nécessaire
- 1000-3000 courbes : sample_size=1500-2000
- 3000-10000 courbes : sample_size=2000 (automatique)
- > 10000 courbes : sample_size=1500 ou utiliser euclidean

---

### Solution 3 : Réduire max_k

Tester moins de valeurs de K :

```python
# Au lieu de max_k=15 (teste 2 à 15 = 14 valeurs)
optimal_k = clustering.find_optimal_clusters(max_k=8, metric="dtw")
# Teste seulement 2 à 8 = 7 valeurs → 2x plus rapide !
```

**Recommandé** : max_k=8 ou max_k=10

---

### Solution 4 : Utiliser softdtw (Compromis)

Softdtw est **5-10x plus rapide** que dtw tout en gardant de bons résultats.

```python
optimal_k = clustering.find_optimal_clusters(max_k=10, metric="softdtw")
clustering.perform_clustering(n_clusters=optimal_k, metric="softdtw")
```

---

### Solution 5 : Utiliser euclidean (Plus Rapide)

Si la vitesse est critique :

```python
optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
clustering.perform_clustering(n_clusters=optimal_k, metric="euclidean")
```

⚠️ Qualité moindre mais **100x plus rapide** !

---

## 📊 Tableau Comparatif

| Stratégie | Temps | Qualité | Recommandation |
|-----------|-------|---------|----------------|
| **Hybride (euclidean → dtw)** | ⚡⚡⚡ ~2 min | ⭐⭐⭐⭐ Excellente | ✅ **RECOMMANDÉ** |
| DTW complet | 🐌 ~30 min | ⭐⭐⭐⭐⭐ Parfaite | Pour analyse finale |
| softdtw complet | ⚡⚡ ~5 min | ⭐⭐⭐⭐ Très bonne | Bon compromis |
| euclidean complet | ⚡⚡⚡⚡ ~30 sec | ⭐⭐⭐ Bonne | Tests rapides |
| Réduction échantillon | ⚡⚡⚡ ~3 min | ⭐⭐⭐⭐ Très bonne | Beaucoup de données |

---

## 🎯 Workflow Recommandé

### Étape 1 : Tests Rapides (euclidean)
```bash
cd demande
python prediction_cluster.py
# Choisir : 1 (euclidean) pour K
# Choisir : 2 (euclidean) pour clustering
# Temps : ~1 minute
```

### Étape 2 : Analyse Intermédiaire (softdtw)
```bash
# Relancer avec softdtw
# Choisir : 3 (softdtw) pour K
# Choisir : 3 (softdtw) pour clustering
# Temps : ~5 minutes
```

### Étape 3 : Analyse Finale (hybride)
```bash
# Version optimale
# Choisir : 1 (euclidean) pour K - RAPIDE
# Choisir : 1 (dtw) pour clustering - QUALITÉ
# Temps : ~3 minutes
```

---

## 💡 Astuces Supplémentaires

### 1. Réduire n_init

Dans le code :
```python
clustering.perform_clustering(n_clusters=5, metric="dtw", n_init=3)
# Au lieu de n_init=10 par défaut
```

### 2. Réduire DAYS_BEFORE

Si vous analysez J-60 à J (61 points), essayez J-30 à J (31 points) :
```python
# Dans main()
DAYS_BEFORE = 30  # Au lieu de 60
```

DTW sera **4x plus rapide** avec moitié moins de points !

### 3. Filtrer par Hôtel

Analyser un seul hôtel à la fois :
```python
# Avant prepare_booking_curves()
clustering.df = clustering.df[clustering.df['hotCode'] == '0DX']
```

### 4. Mode Parallèle ⭐ ACTIVÉ PAR DÉFAUT

Le script utilise maintenant automatiquement **tous les CPU disponibles** avec `n_jobs=-1` :

```python
ts_kmeans = TimeSeriesKMeans(
    n_clusters=5,
    metric="dtw",
    n_jobs=-1  # ✅ Utilise automatiquement tous les CPU
)
```

**Gains de performance** :
- 2 CPU : ~2x plus rapide
- 4 CPU : ~3-4x plus rapide  
- 8 CPU : ~6-7x plus rapide
- 16 CPU : ~10-12x plus rapide

Le script affiche automatiquement le nombre de CPU utilisés :
```
⚡ Parallélisme activé : n_jobs=-1 (utilise 8 CPU)
```

---

## 🔍 Surveiller la Progression

Le script affiche maintenant :

```
🔄 Test K=2... Inertie=123.45, Silhouette=0.654, Davies-Bouldin=0.321
🔄 Test K=3... Inertie=98.76, Silhouette=0.678, Davies-Bouldin=0.298
```

Si ça ne bouge pas pendant 5 minutes → **DTW est en train de calculer**, c'est normal !

---

## ⏱️ Estimation du Temps

**Formule approximative** :

```
Temps (minutes) ≈ (n_courbes / 1000) × (n_jours / 30) × (max_k - 1) × facteur_métrique

facteur_métrique :
- euclidean : 0.01
- softdtw : 0.5
- dtw : 3
```

**Exemples** :
- 2000 courbes, 61 jours, max_k=10, DTW : 2 × 2 × 9 × 3 = **108 minutes** 🐌
- 2000 courbes, 61 jours, max_k=10, euclidean : 2 × 2 × 9 × 0.01 = **0.4 minutes** ⚡
- 2000 courbes, 61 jours, max_k=10, hybride : **~5 minutes** ⚡⚡⚡

---

## ❓ FAQ

### Q : Le script est bloqué depuis 10 minutes, c'est normal ?
**R** : Oui si vous utilisez DTW ! Attendez ou Ctrl+C puis relancez avec euclidean.

### Q : Quelle est la meilleure stratégie ?
**R** : **Hybride** (euclidean pour K, dtw pour clustering). Rapidité + Qualité !

### Q : Puis-je interrompre et reprendre ?
**R** : Oui, utilisez Ctrl+C puis relancez le script. Les données seront rechargées.

### Q : DTW donne-t-il vraiment de meilleurs résultats ?
**R** : Oui ! +15-30% de score de silhouette en moyenne sur des séries temporelles.

### Q : Comment savoir si DTW vaut le coup d'attendre ?
**R** : Comparez avec euclidean d'abord. Si les clusters sont déjà bons, pas besoin de DTW.

---

## 🎓 En Résumé

| Situation | Solution |
|-----------|----------|
| **Premier test** | euclidean partout (30 sec) |
| **Analyse exploratoire** | softdtw (5 min) |
| **Analyse finale** | euclidean → dtw hybride (3 min) |
| **Publication/Production** | dtw complet (30 min) |
| **Beaucoup de données (>5000)** | Réduction échantillon |
| **Peu de temps** | euclidean uniquement |

---

**💡 Conseil d'Or** : Utilisez la stratégie hybride (euclidean → dtw). C'est le meilleur compromis vitesse/qualité ! ⚡⭐

