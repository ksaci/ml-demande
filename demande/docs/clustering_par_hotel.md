# Analyse de Clustering par Hôtel

## Vue d'ensemble

Le script `prediction_cluster.py` a été modifié pour permettre l'analyse de clustering par hôtel individuel. Au lieu d'analyser tous les hôtels dans un seul fichier, vous pouvez maintenant analyser les données d'un hôtel spécifique.

## Modifications apportées

### 1. Structure des fichiers

**Avant :**
```
data/
  └── Indicateurs.csv          # Tous les hôtels
results/
  ├── clustering_model.pkl
  ├── clustering_results.csv
  └── ...
```

**Après :**
```
data/
  ├── D09/
  │   └── Indicateurs.csv      # Données de l'hôtel D09
  ├── A12/
  │   └── Indicateurs.csv      # Données de l'hôtel A12
  └── ...
results/
  ├── D09/                      # Résultats pour D09
  │   ├── clustering_model.pkl
  │   ├── clustering_results.csv
  │   └── ...
  ├── A12/                      # Résultats pour A12
  │   ├── clustering_model.pkl
  │   └── ...
  └── ...
```

### 2. Modifications de la classe `HotelBookingClustering`

#### Nouveau paramètre `hotCode`

Le constructeur accepte maintenant un paramètre `hotCode` :

```python
# Nouvelle méthode (recommandée)
clustering = HotelBookingClustering(hotCode='D09', days_before=60)

# Ancienne méthode (toujours supportée)
clustering = HotelBookingClustering(csv_path='data/custom/file.csv', days_before=60)
```

#### Chemin automatique des fichiers

Lorsque vous spécifiez `hotCode`, les chemins sont automatiquement configurés :
- **Données** : `data/{hotCode}/Indicateurs.csv`
- **Résultats** : `results/{hotCode}/`

### 3. Méthodes de sauvegarde mises à jour

Toutes les méthodes de sauvegarde utilisent maintenant le répertoire spécifique à l'hôtel :

| Méthode | Chemin par défaut |
|---------|------------------|
| `save_model()` | `results/{hotCode}/clustering_model.pkl` |
| `save_results()` | `results/{hotCode}/clustering_results.csv` |
| `save_cluster_profiles()` | `results/{hotCode}/cluster_profiles.csv` |
| Graphiques | `results/{hotCode}/*.png` |

## Utilisation

### Méthode 1 : Via la ligne de commande (recommandée)

```bash
# Exécuter l'analyse pour l'hôtel D09
python prediction_cluster.py D09

# Exécuter pour un autre hôtel
python prediction_cluster.py A12
```

**⚠️ Note importante :** Le code hôtel est **obligatoire** en argument. Si vous exécutez le script sans argument, vous obtiendrez une erreur avec les instructions d'usage.

Ou en appelant `main()` directement :

```python
if __name__ == "__main__":
    main(hotCode='D09')  # Spécifier directement le code
```

### Méthode 2 : Importation directe

```python
from prediction_cluster import HotelBookingClustering

# Créer l'instance pour l'hôtel D09
clustering = HotelBookingClustering(hotCode='D09', days_before=60)

# Charger les données
clustering.load_data(year_filter=2024)

# Préparer les courbes
clustering.prepare_booking_curves(min_observations=20)

# Lissage (optionnel)
clustering.apply_smoothing(enable=True, window_length=15)

# Analyser
clustering.analyze_initial_occupancy()

# Normaliser et clusteriser
clustering.normalize_curves()
optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
clustering.perform_clustering(n_clusters=optimal_k, metric="dtw")

# Visualiser et analyser
clustering.visualize_clusters()
clustering.analyze_cluster_characteristics()

# Sauvegarder
clustering.save_results()
clustering.save_model()
clustering.save_cluster_profiles()
```

### Méthode 3 : Script d'exemple

Utilisez le script d'exemple fourni :

```bash
# Avec argument (recommandé)
python example_clustering_by_hotel.py D09

# Sans argument (demande interactivement)
python example_clustering_by_hotel.py
```

## Chargement d'un modèle sauvegardé

```python
from prediction_cluster import HotelBookingClustering

# Créer l'instance
clustering = HotelBookingClustering(hotCode='D09')

# Charger le modèle sauvegardé
clustering.load_model()  # Charge automatiquement depuis results/D09/clustering_model.pkl

# Ou spécifier un chemin personnalisé
clustering.load_model(model_path='path/to/custom_model.pkl')
```

## Prédiction pour une nouvelle courbe

```python
from prediction_cluster import HotelBookingClustering

# Charger le modèle
clustering = HotelBookingClustering(hotCode='D09')
clustering.load_model()

# Prédire le cluster pour une courbe incomplète
partial_curve = {
    'J-60': 0.10,
    'J-59': 0.12,
    'J-58': 0.13,
    # ... jusqu'à J-15 par exemple
    'J-15': 0.35
}

result = clustering.predict_cluster(partial_curve)

print(f"Cluster prédit : {result['cluster']}")
print(f"Confiance : {result['confidence']:.3f}")
print(f"Distances : {result['all_distances']}")
```

## Exemples d'utilisation

### Exemple 1 : Analyser plusieurs hôtels

```python
from prediction_cluster import HotelBookingClustering

hotels = ['D09', 'A12', 'B05', 'C23']

for hotel_code in hotels:
    print(f"\n{'='*60}")
    print(f"Traitement de l'hôtel {hotel_code}")
    print(f"{'='*60}\n")
    
    clustering = HotelBookingClustering(hotCode=hotel_code, days_before=60)
    clustering.load_data(year_filter=2024)
    clustering.prepare_booking_curves(min_observations=20)
    clustering.apply_smoothing(enable=True)
    clustering.analyze_initial_occupancy()
    clustering.normalize_curves()
    
    optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
    clustering.perform_clustering(n_clusters=optimal_k, metric="dtw", n_init=5)
    
    clustering.visualize_clusters()
    clustering.analyze_cluster_characteristics()
    clustering.save_results()
    clustering.save_model()
    clustering.save_cluster_profiles()
    
    print(f"\n✓ Hôtel {hotel_code} terminé !")
```

### Exemple 2 : Comparer les profils de plusieurs hôtels

```python
import pandas as pd
from prediction_cluster import HotelBookingClustering

hotels = ['D09', 'A12', 'B05']

for hotel_code in hotels:
    clustering = HotelBookingClustering(hotCode=hotel_code)
    clustering.load_model()
    
    # Charger les profils de clusters
    profiles = pd.read_csv(f'results/{hotel_code}/cluster_profiles.csv', sep=';')
    
    print(f"\n🏨 Hôtel {hotel_code} - {len(profiles)} clusters")
    print(profiles[['cluster', 'n_samples', 'percentage']].to_string(index=False))
```

## Migration depuis l'ancienne version

Si vous avez des scripts qui utilisent l'ancienne version, ils continueront à fonctionner :

```python
# Ancienne méthode (toujours supportée)
clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
```

Pour migrer vers la nouvelle méthode :

```python
# Nouvelle méthode
clustering = HotelBookingClustering(hotCode='D09', days_before=60)
```

## Rétrocompatibilité

- ✅ Les anciens modèles peuvent être chargés (le `hotCode` est optionnel)
- ✅ Le paramètre `csv_path` est toujours supporté
- ✅ Les chemins personnalisés fonctionnent dans toutes les méthodes `save_*()`
- ✅ Le comportement par défaut reste le même si vous ne spécifiez pas `hotCode`

## Fichiers générés

Pour chaque hôtel analysé, les fichiers suivants sont créés dans `results/{hotCode}/` :

| Fichier | Description |
|---------|-------------|
| `clustering_model.pkl` | Modèle de clustering sauvegardé (KMeans + scaler) |
| `clustering_results.csv` | Résultats détaillés avec clusters assignés |
| `cluster_profiles.csv` | Profils moyens de chaque cluster |
| `initial_occupancy_analysis.png` | Analyse exploratoire des taux d'occupation |
| `clustering_optimal_k.png` | Graphique pour déterminer le K optimal |
| `clustering_curves_by_cluster.png` | Courbes regroupées par cluster |
| `clustering_comparison.png` | Comparaison des profils moyens |
| `clustering_pca.png` | Projection PCA en 2D des clusters |

## Notes importantes

1. **Structure des dossiers** : Assurez-vous que la structure `data/{hotCode}/Indicateurs.csv` existe avant d'exécuter l'analyse.

2. **Code hôtel** : Le code hôtel doit être de 3 caractères (ex: `D09`, `A12`). Une validation est effectuée automatiquement.

3. **Création automatique** : Le dossier `results/{hotCode}/` est créé automatiquement s'il n'existe pas.

4. **Filtrage automatique** : Lorsque vous utilisez `hotCode`, seules les données de cet hôtel sont analysées (pas de filtrage manuel nécessaire).

## Support

Pour toute question ou problème :
- Consultez la documentation du module principal : `clustering_prediction_guide.md`
- Vérifiez que la structure des fichiers est correcte
- Vérifiez que le fichier `data/{hotCode}/Indicateurs.csv` existe et contient des données valides

