# Guide Rapide - Clustering par Hôtel

## 🚀 Utilisation rapide

### Exécuter l'analyse pour un hôtel

```bash
python prediction_cluster.py D09
```

**Prérequis :**
- Le fichier `data/D09/Indicateurs.csv` doit exister
- Format du code hôtel : 3 caractères (ex: D09, A12, B05)

### Résultats générés

Tous les fichiers sont sauvegardés dans `results/{hotCode}/` :

```
results/D09/
├── clustering_model.pkl                    # Modèle sauvegardé
├── clustering_results.csv                  # Résultats détaillés
├── cluster_profiles.csv                    # Profils moyens
├── initial_occupancy_analysis.png          # Analyse exploratoire
├── clustering_optimal_k.png                # Détermination du K
├── clustering_curves_by_cluster.png        # Courbes par cluster
├── clustering_comparison.png               # Comparaison des profils
└── clustering_pca.png                      # Projection PCA
```

## 📊 Exemples d'utilisation

### 1. Analyser un hôtel

```bash
python prediction_cluster.py D09
```

### 2. Analyser plusieurs hôtels

```bash
# Créer un script batch (Windows)
for hotel in D09 A12 B05 C23
do
    python prediction_cluster.py $hotel
done
```

Ou en Python :

```python
import subprocess

hotels = ['D09', 'A12', 'B05', 'C23']

for hotel in hotels:
    print(f"\n{'='*60}")
    print(f"Traitement de l'hôtel {hotel}")
    print(f"{'='*60}\n")
    
    subprocess.run(['python', 'prediction_cluster.py', hotel])
```

### 3. Utiliser le modèle sauvegardé

```python
from prediction_cluster import HotelBookingClustering

# Charger le modèle
clustering = HotelBookingClustering(hotCode='D09')
clustering.load_model()

# Prédire pour une nouvelle courbe
partial_curve = {
    'J-60': 0.10,
    'J-59': 0.12,
    'J-50': 0.20,
    'J-40': 0.28,
    'J-30': 0.35,
    'J-20': 0.45,
    'J-15': 0.52
}

result = clustering.predict_cluster(partial_curve)
print(f"Cluster prédit : {result['cluster']}")
print(f"Confiance : {result['confidence']:.3f}")
```

## ⚙️ Configuration

Par défaut, l'analyse utilise :

- **Période analysée :** J-60 à J (60 jours avant le séjour)
- **Années :** Toutes les années disponibles
- **Lissage :** Activé (Savitzky-Golay, fenêtre=15)
- **Nombre de clusters :** 10 (fixe)
- **Recherche K optimal :** DÉSACTIVÉE (pour gagner du temps)
- **Métrique clustering :** EUCLIDEAN (mode développement - rapide)

### Modifier la configuration

Éditez la fonction `main()` dans `prediction_cluster.py` :

```python
# Options de clustering
N_CLUSTERS = 10  # Nombre de clusters (par défaut : 10)
AUTO_FIND_K = False  # Recherche automatique du nombre optimal (True pour activer)
USE_DTW = False  # False = Euclidean (rapide) | True = DTW (qualité)
```

**Mode développement (rapide) :**
```python
USE_DTW = False  # Utilise Euclidean (~2-3x plus rapide)
```

**Mode production (meilleure qualité) :**
```python
USE_DTW = True  # Utilise DTW (meilleure qualité mais plus lent)
```

**Recherche automatique du K optimal :**
```python
AUTO_FIND_K = True  # Active la recherche (plus lent)
```

## 📁 Structure des dossiers requise

```
demande/
├── data/
│   ├── D09/
│   │   └── Indicateurs.csv      ← Vos données
│   ├── A12/
│   │   └── Indicateurs.csv
│   └── ...
├── results/                      ← Créé automatiquement
│   ├── D09/
│   ├── A12/
│   └── ...
├── prediction_cluster.py         ← Script principal
└── example_clustering_by_hotel.py
```

## ❓ Aide

### Erreur : "Code hôtel manquant"

```bash
❌ ERREUR : Code hôtel manquant !

Usage:
  python prediction_cluster.py <hotCode>

Exemple:
  python prediction_cluster.py D09
```

**Solution :** Ajoutez le code hôtel en argument : `python prediction_cluster.py D09`

### Erreur : "FileNotFoundError"

```
FileNotFoundError: data/D09/Indicateurs.csv
```

**Solution :** Vérifiez que le fichier existe et que le chemin est correct.

### Erreur : "Aucune donnée pour l'année"

```
⚠️ ATTENTION : Aucune donnée pour l'année 2024 !
Années disponibles : [2022, 2023]
```

**Solution :** Modifiez `YEAR_FILTER` dans la fonction `main()` ou utilisez `None` pour toutes les années.

## 📚 Documentation complète

- **Guide complet :** `docs/clustering_par_hotel.md`
- **Guide de prédiction :** `docs/clustering_prediction_guide.md`

## 💡 Astuces

### Performance

- **DTW est lent** sur beaucoup de données (>5000 courbes)
- La recherche du K optimal utilise `euclidean` (rapide)
- Le clustering final utilise `DTW` (meilleure qualité)
- Sur Windows, le parallélisme est désactivé par défaut pour éviter les erreurs

### Qualité

- **Lissage recommandé** pour réduire le bruit
- **Minimum 20 observations** par courbe (paramétrable)
- **Interpolation linéaire** pour les valeurs manquantes

### Interprétation

- Les clusters représentent des **profils de réservation**
- Exemple : "Dernière minute", "Anticipé", "Régulier", etc.
- Consultez le fichier `cluster_profiles.csv` pour les statistiques détaillées
