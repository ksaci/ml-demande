# 📚 Guide Complet - Système XGBoost de Prédiction TO

## 🎯 Introduction

Ce guide vous accompagne de l'installation à l'utilisation en production du système de prédiction XGBoost.

---

## 📦 Fichiers Créés

### Scripts Python (5 fichiers)

| Fichier | Description | Utilisation |
|---------|-------------|-------------|
| `xgboost_train_model.py` | Script principal d'entraînement | `python xgboost_train_model.py` |
| `xgboost_predict_example.py` | Exemple de prédiction | `python xgboost_predict_example.py` |
| `test_xgboost_setup.py` | Validation de l'environnement | `python test_xgboost_setup.py` |
| `load_model_from_azure.py` | Gestion modèles Azure | `python load_model_from_azure.py --list` |

### Documentation (4 fichiers)

| Fichier | Contenu |
|---------|---------|
| `docs/XGBOOST_TRAINING_DOC.md` | Documentation technique complète (600+ lignes) |
| `README_XGBOOST.md` | Guide Quick Start |
| `RECAP_XGBOOST.md` | Résumé des fonctionnalités |
| `GUIDE_COMPLET_XGBOOST.md` | Ce fichier |

### Configuration (2 fichiers)

| Fichier | Description |
|---------|-------------|
| `config_xgboost.yaml` | Configuration paramétrable |
| `requirements_xgboost.txt` | Dépendances Python |

---

## 🚀 Installation Complète

### Étape 1 : Installer les Dépendances

```bash
cd demande

# Installation des packages Python
pip install -r requirements_xgboost.txt
```

### Étape 2 : Configurer Azure (Optionnel)

```bash
# Option 1 : Variable d'environnement
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=VOTRE_COMPTE;AccountKey=VOTRE_CLE;EndpointSuffix=core.windows.net"

# Option 2 : Fichier .env (nécessite python-dotenv)
echo 'AZURE_STORAGE_CONNECTION_STRING="..."' > .env
```

### Étape 3 : Vérifier l'Installation

```bash
python test_xgboost_setup.py
```

**Sortie attendue:**
```
✅ CONFIGURATION VALIDE
   Vous pouvez lancer l'entraînement avec:
   python xgboost_train_model.py
```

---

## 📊 Workflow Complet

### 1️⃣ Entraînement Initial

```bash
# Avec configuration par défaut
python xgboost_train_model.py

# Avec configuration personnalisée
python xgboost_train_model.py --config ma_config.yaml

# Sans sauvegarde Azure
python xgboost_train_model.py --no-azure
```

**Résultats générés:**
- `results/models/xgb_to_predictor.joblib`
- `results/models/xgb_scaler.joblib`
- `results/models/feature_columns.txt`
- `results/xgb_scatter_plot.png`
- `results/xgb_feature_importance.png`
- `xgboost_training.log`

### 2️⃣ Vérification des Performances

```bash
# Vérifier les logs
tail -f xgboost_training.log

# Ouvrir les graphiques
open results/xgb_scatter_plot.png
open results/xgb_feature_importance.png
```

**Métriques dans les logs:**
```
📊 MÉTRIQUES FINALES:
   Train MAE: 0.0450
   Train R²:  0.8900
   Test MAE:  0.0560
   Test R²:   0.8265
```

### 3️⃣ Utilisation du Modèle Local

```bash
# Exemple de prédiction
python xgboost_predict_example.py
```

**Sortie:**
```
✅ PRÉDICTION : TO final = 0.7234 (72.34%)

📈 Analyse:
   TO actuel (J-8): 0.6900 (69.00%)
   TO prédit (J-0): 0.7234 (72.34%)
   Évolution: +0.0334 (+4.84%)
   📊 Tendance: Montée attendue
```

### 4️⃣ Gestion Azure Blob Storage

```bash
# Lister les modèles disponibles
python load_model_from_azure.py --list

# Télécharger le dernier modèle
python load_model_from_azure.py --download latest

# Télécharger un modèle spécifique
python load_model_from_azure.py --download 20241216_143025

# Télécharger dans un répertoire personnalisé
python load_model_from_azure.py --download latest --output mon_dossier
```

---

## ⚙️ Configuration Avancée

### Personnalisation via YAML

**Éditez `config_xgboost.yaml`:**

```yaml
# Modifier les hyperparamètres
model:
  n_estimators: 800        # Plus d'arbres
  learning_rate: 0.03      # Apprentissage plus lent
  max_depth: 9             # Arbres plus profonds

# Changer l'horizon de prédiction
prediction:
  horizon: 14              # Prédire à J+14

# Désactiver Azure
azure:
  save_to_blob: false
```

**Puis:**
```bash
python xgboost_train_model.py --config config_xgboost.yaml
```

### Personnalisation par Code

```python
from xgboost_train_model import XGBoostOccupancyPredictor

config = {
    'clustering_results_path': 'mes_donnees/clustering.csv',
    'indicateurs_path': 'mes_donnees/indicateurs.csv',
    'prediction_horizon': 14,
    'model_params': {
        'n_estimators': 1000,
        'max_depth': 10,
        # ...
    }
}

predictor = XGBoostOccupancyPredictor(config)
# ... pipeline complet
```

---

## 🔍 Utilisation en Production

### Scénario 1 : Prédiction Unique

```python
import joblib
from xgboost_predict_example import compute_pm_features

# Charger le modèle
model = joblib.load("results/models/xgb_to_predictor.joblib")
scaler = joblib.load("results/models/xgb_scaler.joblib")

# Préparer les données
to_series = [0.05, 0.06, ..., 0.69]  # 53 valeurs
pm_series = [120, 121, ..., 125]     # 53 valeurs

# Créer le vecteur de features
row_dict = {}
for i, val in enumerate(range(60, 7, -1)):
    row_dict[f"J-{val}"] = to_series[i]

row_dict.update(compute_pm_features(pm_series))
row_dict.update({
    "cluster": 3,
    "month": 8,
    "dayofweek": 4,
    "nb_observations": 53
})

# Prédire
row_df = pd.DataFrame([row_dict])
row_scaled = scaler.transform(row_df)
prediction = model.predict(row_scaled)[0]
```

### Scénario 2 : Prédictions en Batch

```python
import pandas as pd

# Charger un fichier de nouvelles données
new_data = pd.read_csv("nouvelles_donnees.csv")

# Appliquer le pipeline de préparation
predictor = XGBoostOccupancyPredictor(config)
X_new = predictor.prepare_features(new_data)

# Normaliser
X_new_scaled = predictor.scaler.transform(X_new)

# Prédire
predictions = predictor.model.predict(X_new_scaled)

# Sauvegarder les résultats
results_df = pd.DataFrame({
    'stay_date': new_data['stay_date'],
    'predicted_to': predictions
})
results_df.to_csv("predictions.csv", index=False)
```

### Scénario 3 : API REST (Flask)

```python
from flask import Flask, request, jsonify
from xgboost_predict_example import load_model_artifacts, predict_to

app = Flask(__name__)

# Charger le modèle au démarrage
model, scaler, feature_cols = load_model_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    prediction = predict_to(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        to_series=data['to_series'],
        pm_series=data['pm_series'],
        cluster=data['cluster'],
        month=data['month'],
        dayofweek=data['dayofweek']
    )
    
    return jsonify({'predicted_to': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 🔄 Workflow de Réentraînement

### Quand Réentraîner ?

- ✅ **Mensuel** : Routine recommandée
- ⚠️ **Ad-hoc** si :
  - Nouvelles données > 10% du volume actuel
  - Performance dégradée (MAE > 0.07)
  - Changement de saisonnalité
  - Feedback métier

### Procédure de Réentraînement

```bash
# 1. Vérifier les nouvelles données
python test_xgboost_setup.py

# 2. Lancer l'entraînement
python xgboost_train_model.py

# 3. Comparer les performances
# Ancienne version
echo "Ancien modèle: MAE = 0.0560, R² = 0.8265"

# Nouvelle version (dans les logs)
tail xgboost_training.log

# 4. Si meilleur : déployer
python load_model_from_azure.py --download latest

# 5. Tester en production
python xgboost_predict_example.py
```

---

## 🛠️ Maintenance

### Monitoring des Performances

Créez un fichier `monitor_model.py`:

```python
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

# Charger le modèle
model = joblib.load("results/models/xgb_to_predictor.joblib")
scaler = joblib.load("results/models/xgb_scaler.joblib")

# Charger les vraies valeurs vs prédictions
real_data = pd.read_csv("production_data.csv")

# Comparer
mae = mean_absolute_error(real_data['real_to'], real_data['predicted_to'])

# Alerter si performance dégradée
if mae > 0.07:
    print(f"⚠️  ALERTE : Performance dégradée (MAE = {mae:.4f})")
    # Envoyer une notification...
```

### Optimisation des Hyperparamètres

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [400, 600, 800],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"Meilleurs params: {grid_search.best_params_}")
```

---

## 📊 Analyse des Résultats

### Interpréter le Scatter Plot

```
TO Réel vs TO Prédit
│
│   Points au-dessus de la ligne → Sous-estimation
│   Points en-dessous → Sur-estimation
│   Points sur la ligne → Prédiction parfaite
│
└─────────────────────────────────────────────
```

### Interpréter Feature Importance

- **J-8 (46%)** : Le TO à J-8 est le meilleur prédicteur
- **J-9 (13%)** : Confirmation de la tendance récente
- **cluster (8%)** : Le comportement type du groupe
- **Features PM (~3%)** : Impact du prix moyen

➡️ **Conclusion:** Les 7-8 derniers jours de TO sont critiques

---

## 🐛 Troubleshooting Complet

### Problème : "ModuleNotFoundError"

```bash
# Réinstaller toutes les dépendances
pip install -r requirements_xgboost.txt --force-reinstall
```

### Problème : "Fichier non trouvé"

```bash
# Vérifier la structure
python test_xgboost_setup.py

# Vérifier manuellement
ls -la results/clustering_results.csv
ls -la data/Indicateurs.csv
```

### Problème : "Azure Blob Error"

```bash
# Test de connexion
python -c "from azure.storage.blob import BlobServiceClient; print('✅ Azure OK')"

# Vérifier la connection string
env | grep AZURE

# Tester sans Azure
python xgboost_train_model.py --no-azure
```

### Problème : "Mauvaise Performance (R² < 0.70)"

**Diagnostic:**
1. Vérifier la qualité des données
2. Analyser les outliers
3. Vérifier la distribution train/test
4. Augmenter `n_estimators`

**Actions:**
```python
# Analyser les résidus
residuals = y_test - y_pred
plt.hist(residuals, bins=50)
plt.show()

# Identifier les pires prédictions
worst = pd.DataFrame({
    'real': y_test,
    'pred': y_pred,
    'error': abs(y_test - y_pred)
}).sort_values('error', ascending=False).head(10)
```

---

## 📈 Cas d'Usage Pratiques

### Use Case 1 : Prédiction pour demain

```python
from xgboost_predict_example import load_model_artifacts, predict_to

# Charger le modèle
model, scaler, features = load_model_artifacts()

# Données du jour
to_aujourd_hui = [...]  # TO de J-60 à J-8
pm_aujourd_hui = [...]  # PM de J-60 à J-8

# Prédiction
to_predit = predict_to(
    model, scaler, features,
    to_aujourd_hui, pm_aujourd_hui,
    cluster=3, month=12, dayofweek=2
)

print(f"TO prédit pour demain: {to_predit*100:.1f}%")
```

### Use Case 2 : Analyse de Sensibilité

```python
# Test de différents clusters
for cluster in range(7):
    pred = predict_to(..., cluster=cluster, ...)
    print(f"Cluster {cluster}: {pred:.3f}")

# Test de différents mois
for month in range(1, 13):
    pred = predict_to(..., month=month, ...)
    print(f"Mois {month}: {pred:.3f}")
```

### Use Case 3 : Batch Processing

```python
# Charger plusieurs réservations
reservations = pd.read_csv("reservations_to_predict.csv")

predictions = []
for idx, row in reservations.iterrows():
    pred = predict_to(
        model, scaler, features,
        row['to_series'], row['pm_series'],
        row['cluster'], row['month'], row['dayofweek']
    )
    predictions.append(pred)

reservations['predicted_to'] = predictions
reservations.to_csv("predictions_batch.csv")
```

---

## 🔐 Sécurité et Bonnes Pratiques

### Variables d'Environnement

**Ne JAMAIS commit:**
- `.env`
- Connection strings
- Clés API

**À faire:**
```bash
# Ajouter au .gitignore
echo ".env" >> .gitignore
echo "*.log" >> .gitignore
```

### Gestion des Versions

**Versionner:**
- ✅ Scripts Python
- ✅ Configuration YAML
- ✅ Documentation
- ✅ Requirements.txt

**Ne PAS versionner:**
- ❌ Modèles (.joblib)
- ❌ Logs
- ❌ Fichiers temporaires
- ❌ Credentials

### Backup

```bash
# Sauvegarder périodiquement
cp results/models/*.joblib backups/$(date +%Y%m%d)/
```

---

## 📅 Planning de Maintenance

### Hebdomadaire
- [ ] Vérifier les logs (`xgboost_training.log`)
- [ ] Monitorer les performances en production

### Mensuel
- [ ] Réentraîner le modèle
- [ ] Comparer avec version précédente
- [ ] Mettre à jour la documentation si changements

### Trimestriel
- [ ] Audit complet du code
- [ ] Optimisation des hyperparamètres
- [ ] Revue des features (ajout/suppression)

---

## 🎓 Formation de l'Équipe

### Pour les Data Scientists

**Lire:**
1. `docs/XGBOOST_TRAINING_DOC.md` (technique)
2. Code source de `xgboost_train_model.py`
3. Notebook original `test_xgboost_prediction.ipynb`

**Pratiquer:**
1. Lancer un entraînement complet
2. Modifier les hyperparamètres
3. Ajouter une nouvelle feature

### Pour les DevOps

**Lire:**
1. `README_XGBOOST.md` (deployment)
2. Section Azure de la doc technique

**Pratiquer:**
1. Configurer Azure Blob Storage
2. Automatiser le réentraînement (cron/airflow)
3. Mettre en place le monitoring

### Pour les Utilisateurs Métier

**Lire:**
1. `RECAP_XGBOOST.md` (overview)
2. Section "Métriques" de la doc technique

**Utiliser:**
1. `xgboost_predict_example.py` pour tester
2. Interpréter les résultats (MAE, R²)

---

## 🎯 Checklist de Déploiement

### Avant le Premier Lancement

- [ ] Installation des dépendances vérifiée
- [ ] Fichiers de données présents et validés
- [ ] Configuration Azure testée
- [ ] `test_xgboost_setup.py` exécuté avec succès

### Après l'Entraînement

- [ ] Logs vérifiés (pas d'erreurs)
- [ ] Métriques satisfaisantes (R² > 0.75)
- [ ] Graphiques générés et cohérents
- [ ] Modèle sauvegardé (local + Azure)

### Avant la Production

- [ ] Test de prédiction réussi
- [ ] Validation métier des prédictions
- [ ] Documentation à jour
- [ ] Plan de monitoring en place

---

## 📚 Références

### Documentation Interne
- 📖 [Documentation Technique Complète](docs/XGBOOST_TRAINING_DOC.md)
- 📄 [Quick Start Guide](README_XGBOOST.md)
- 📝 [Résumé des Fonctionnalités](RECAP_XGBOOST.md)

### Documentation Externe
- [XGBoost Official Docs](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Azure Blob Storage Python SDK](https://docs.microsoft.com/azure/storage/blobs/storage-quickstart-blobs-python)

---

## 💡 Astuces et Conseils

### Performance

```python
# Utiliser tous les CPU
'n_jobs': -1

# Augmenter le learning rate si overfitting
'learning_rate': 0.1

# Régularisation plus forte
'reg_lambda': 2.0
```

### Debugging

```python
# Activer le mode verbose
import logging
logging.basicConfig(level=logging.DEBUG)

# Vérifier les features
print(model.get_booster().feature_names)

# Analyser les prédictions
print(f"Min: {y_pred.min()}, Max: {y_pred.max()}")
```

### Optimisation

```python
# Cross-validation pour validation robuste
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='neg_mean_absolute_error'
)
print(f"CV MAE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 🎉 Conclusion

Vous avez maintenant un système complet et professionnel pour :
- ✅ Entraîner un modèle XGBoost de prédiction TO
- ✅ Sauvegarder dans Azure Blob Storage
- ✅ Faire des prédictions en production
- ✅ Monitorer et maintenir le système

**Bon travail ! 🚀**

---

**Version:** 1.0  
**Dernière mise à jour:** Décembre 2024  
**Contact:** Équipe Data Science

