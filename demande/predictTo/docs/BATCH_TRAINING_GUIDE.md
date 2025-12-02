# Guide d'utilisation du script batch `run_predictTo_batch.py`

## 📋 Vue d'ensemble

Le script `run_predictTo_batch.py` permet d'entraîner automatiquement des modèles XGBoost pour un hôtel donné avec plusieurs horizons de prédiction en une seule commande.

## 🎯 Horizons par défaut

Le script entraîne des modèles pour les horizons suivants :
- **J-59** : Prédiction 59 jours à l'avance (maximum possible)
- **J-45** : Prédiction 45 jours à l'avance
- **J-30** : Prédiction 30 jours à l'avance
- **J-21** : Prédiction 21 jours à l'avance
- **J-15** : Prédiction 15 jours à l'avance
- **J-10** : Prédiction 10 jours à l'avance
- **J-7** : Prédiction 7 jours à l'avance
- **J-5** : Prédiction 5 jours à l'avance
- **J-3** : Prédiction 3 jours à l'avance
- **J-1** : Prédiction 1 jour à l'avance
- **J-0** : Prédiction le jour même

**Note** : L'horizon maximum est J-59 car les données PM/TO vont jusqu'à J-60. Pour prédire à J-59, on utilise les données de J-60.

## 🚀 Utilisation

### Commande de base

```bash
cd predictTo
python run_predictTo_batch.py --hotel D09
```

Cette commande va :
1. Entraîner 11 modèles (un pour chaque horizon : J-59, J-45, J-30, J-21, J-15, J-10, J-7, J-5, J-3, J-1, J-0)
2. Sauvegarder localement dans `results/D09/{hotel}/J-{horizon}/`
3. Uploader dans Azure Blob Storage : `ml-models/predictTo/{hotel}/J-{horizon}/`

### Options disponibles

#### 1. Spécifier un hôtel (obligatoire)
```bash
python run_predictTo_batch.py --hotel D09
python run_predictTo_batch.py --hotel 6N8
```

#### 2. Entraîner uniquement certains horizons
```bash
# Seulement J-7, J-14 et J-30
python run_predictTo_batch.py --hotel D09 --horizons 7 14 30

# Seulement les horizons courts
python run_predictTo_batch.py --hotel D09 --horizons 1 3 5 7
```

#### 3. Désactiver la sauvegarde Azure
```bash
python run_predictTo_batch.py --hotel D09 --no-azure
```

#### 4. Activer la recherche d'hyperparamètres
```bash
# ⚠️ ATTENTION : cela va multiplier le temps d'entraînement par ~10-15x
python run_predictTo_batch.py --hotel D09 --search-hyperparams
```

#### 5. Utiliser un fichier de configuration personnalisé
```bash
python run_predictTo_batch.py --hotel D09 --config config_custom.yaml
```

### Combinaisons d'options

```bash
# Entraînement rapide sans Azure pour horizons courts
python run_predictTo_batch.py --hotel D09 --horizons 1 3 5 7 --no-azure

# Entraînement complet avec recherche d'hyperparamètres
python run_predictTo_batch.py --hotel 6N8 --search-hyperparams

# Test sur un seul horizon
python run_predictTo_batch.py --hotel D09 --horizons 7
```

## 📁 Structure de sortie

### Locale
```
results/D09/
└── D09/                    # Code de l'hôtel
    ├── J-60/
    │   ├── models/
    │   │   ├── xgb_to_predictor.joblib
    │   │   ├── xgb_scaler.joblib
    │   │   └── feature_columns.txt
    │   ├── xgb_scatter_plot.png
    │   ├── xgb_feature_importance.png
    │   ├── training_data_before_scaling.csv
    │   └── test_predictions.csv
    ├── J-45/
    ├── J-30/
    ├── J-21/
    ├── J-15/
    ├── J-10/
    ├── J-7/
    ├── J-5/
    ├── J-3/
    ├── J-1/
    └── J-0/
```

### Azure Blob Storage (container `ml-models`)
```
ml-models/
└── predictTo/
    └── D09/                # Code de l'hôtel
    ├── J-59/
    ├── J-45/
    ├── J-30/
    ├── J-21/
    ├── J-15/
    ├── J-10/
    ├── J-7/
    ├── J-5/
    ├── J-3/
    ├── J-1/
    └── J-0/
```

## 📊 Résumé de l'exécution

À la fin de l'exécution, le script affiche un résumé détaillé :

```
================================================================================
📊 RÉSUMÉ DU BATCH TRAINING
================================================================================
Hôtel: D09
Total de modèles: 11
✅ Succès: 11
❌ Erreurs: 0
⏱️  Durée totale: 45.32 minutes (2719.20 secondes)

Détails par horizon:
--------------------------------------------------------------------------------
Horizon    Statut       Durée (s)    Test MAE     Test R²     
--------------------------------------------------------------------------------
J-59       ✅ Succès    245.32       0.0234       0.8567      
J-45       ✅ Succès    238.45       0.0228       0.8612      
J-30       ✅ Succès    242.11       0.0221       0.8701      
J-21       ✅ Succès    239.87       0.0215       0.8765      
J-15       ✅ Succès    241.23       0.0208       0.8823      
J-10       ✅ Succès    243.56       0.0201       0.8891      
J-7        ✅ Succès    240.34       0.0195       0.8945      
J-5        ✅ Succès    238.92       0.0189       0.9012      
J-3        ✅ Succès    241.67       0.0183       0.9078      
J-1        ✅ Succès    239.11       0.0177       0.9145      
J-0        ✅ Succès    238.45       0.0171       0.9201      
--------------------------------------------------------------------------------

================================================================================
✅ BATCH TRAINING TERMINÉ AVEC SUCCÈS
================================================================================
```

## 📝 Logs

Les logs sont sauvegardés dans deux fichiers :
- `predictTo_batch.log` : Log du script batch principal
- `predictTo_training.log` : Logs détaillés de chaque entraînement

## ⏱️ Temps d'exécution estimé

### Sans recherche d'hyperparamètres (par défaut)
- **Par horizon** : ~4-5 minutes
- **11 horizons** : ~45-55 minutes
- **Avec Azure** : +1-2 minutes

### Avec recherche d'hyperparamètres (`--search-hyperparams`)
- **Par horizon** : ~30-45 minutes
- **11 horizons** : ~5-8 heures ⚠️

## 🔧 Prérequis

1. **Données requises** :
   - Résultats de clustering : `../cluster/results/{hotel}/clustering_results.csv`
     - Le script charge automatiquement les données depuis `cluster/results/{hotel}/` quand `--hotel` est spécifié
     - Exemple pour D09 : `../cluster/results/D09/clustering_results.csv`
   - Indicateurs : `../data/D09/Indicateurs.csv`
   - Prix concurrents : `../data/D09/rateShopper.csv`

2. **Configuration Azure** (optionnel) :
   - Variable d'environnement : `AZURE_STORAGE_CONNECTION_STRING`
   - Container : `ml-models`

3. **Dépendances Python** :
   ```bash
   pip install -r requirements_predictTo.txt
   ```

## ❌ Gestion des erreurs

Si un horizon échoue :
- Le script continue avec les autres horizons
- L'erreur est loggée dans le fichier de log
- Le résumé final indique les horizons en erreur

Exemple avec erreurs :
```
⚠️  ERREURS DÉTAILLÉES:
--------------------------------------------------------------------------------
Horizon J-45: FileNotFoundError: Data file not found
Horizon J-30: ValueError: Invalid data format
--------------------------------------------------------------------------------

⚠️  BATCH TRAINING TERMINÉ AVEC 2 ERREUR(S)
```

## 💡 Conseils d'utilisation

1. **Test rapide** : Commencez par un seul horizon pour vérifier que tout fonctionne
   ```bash
   python run_predictTo_batch.py --hotel D09 --horizons 7 --no-azure
   ```

2. **Production** : Utilisez les horizons par défaut avec Azure
   ```bash
   python run_predictTo_batch.py --hotel D09
   ```

3. **Optimisation** : Lancez la recherche d'hyperparamètres sur un horizon représentatif (J-7 ou J-14), puis utilisez les meilleurs paramètres dans `config_predictTo.yaml` pour tous les horizons
   ```bash
   # Étape 1 : Recherche sur J-7
   python predictTo_train_model.py --hotel D09 --horizon 7 --search-hyperparams
   
   # Étape 2 : Mettre à jour config_predictTo.yaml avec les meilleurs params
   
   # Étape 3 : Entraîner tous les horizons avec les params optimisés
   python run_predictTo_batch.py --hotel D09
   ```

## 🔗 Voir aussi

- [Guide complet PredictTO](GUIDE_COMPLET_PREDICTTO.md)
- [Documentation de l'entraînement](PREDICTTO_TRAINING_DOC.md)
- [Documentation des features](features_documentation.md)

