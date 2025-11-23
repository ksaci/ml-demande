# Analyse de Clustering - Hôtels

Analyse de clustering pour les courbes de montée en charge du taux d'occupation des hôtels.

## 🚀 Démarrage rapide

```bash
python prediction_cluster.py D09
```

## 📚 Documentation

Toute la documentation est disponible dans le dossier **`docs/`** :

### Guides principaux
- **[Guide Rapide](docs/README_CLUSTERING.md)** - Utilisation rapide et exemples
- **[Guide Complet](docs/clustering_par_hotel.md)** - Documentation complète
- **[Guide de Prédiction](docs/clustering_prediction_guide.md)** - Utiliser les modèles

### Configuration et optimisation
- **[Mode Développement](docs/MODE_DEVELOPPEMENT.md)** - Configuration dev vs prod
- **[Modification des Clusters](docs/MODIFICATION_CLUSTERS.md)** - Ajuster le nombre de clusters
- **[Changelog](docs/CHANGELOG_CLUSTERING.md)** - Historique des versions

### Autres guides
- **[Performance](docs/GUIDE_PERFORMANCE.md)** - Optimisation des performances
- **[Lissage](docs/GUIDE_SMOOTHING.md)** - Options de lissage des courbes
- **[Parallélisme Windows](docs/FIX_WINDOWS_PARALLEL.md)** - Résoudre les problèmes Windows

## 📊 Configuration actuelle

| Paramètre | Valeur |
|-----------|--------|
| Nombre de clusters | 10 (fixe) |
| Métrique | EUCLIDEAN (mode dev) |
| Recherche auto K | Désactivée |

## 🔧 Scripts disponibles

- `prediction_cluster.py` - Script principal d'analyse
- `run_clustering_batch.py` - Analyse en batch (plusieurs hôtels)
- `example_clustering_by_hotel.py` - Exemple d'utilisation

## 📁 Structure

```
demande/
├── data/{hotCode}/Indicateurs.csv     # Données d'entrée
├── results/{hotCode}/                  # Résultats par hôtel
├── docs/                               # Documentation complète
├── prediction_cluster.py               # Script principal
└── README.md                           # Ce fichier
```

## 💡 Voir aussi

- **[PredictTo](docs/README_PREDICTTO.md)** - Prédiction du taux d'occupation
- **[XGBoost](docs/README_XGBOOST_PREDICTION.md)** - Modèles XGBoost

