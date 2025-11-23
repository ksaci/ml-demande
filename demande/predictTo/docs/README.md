# 🎯 PredictTO - Prédiction du Taux d'Occupation

Système complet de Machine Learning pour prédire le taux d'occupation (TO) à J+7 en utilisant XGBoost.

---

## 📂 Structure du Projet

```
demande/
├── predictTo/                               ← VOUS ÊTES ICI
│   ├── predictTo_train_model.py            # Script principal d'entraînement
│   ├── predictTo_predict_example.py        # Exemple d'utilisation
│   ├── test_predictTo_setup.py             # Validation environnement
│   ├── load_predictTo_from_azure.py        # Gestion modèles Azure
│   ├── config_predictTo.yaml               # Configuration
│   ├── requirements_predictTo.txt          # Dépendances
│   ├── docs/                               # Documentation
│   │   ├── README.md                       # Ce fichier - Vue d'ensemble
│   │   ├── PREDICTTO_TRAINING_DOC.md       # Documentation technique
│   │   └── GUIDE_COMPLET_PREDICTTO.md      # Guide complet
│   └── predictTo_training.log              # Logs (généré)
├── data/
│   └── Indicateurs.csv                     # Données PM/RevPAR
└── results/
    ├── clustering_results.csv              # Résultats clustering
    └── models/                             # Modèles sauvegardés (généré)
        ├── xgb_to_predictor.joblib
        ├── xgb_scaler.joblib
        └── feature_columns.txt
```

---

## 🚀 Démarrage Rapide (5 minutes)

### 1. Installation

```bash
# Se placer dans ce dossier
cd demande/predictTo

# Installer les dépendances
pip install -r requirements_predictTo.txt
```

### 2. Test de Configuration

```bash
python test_predictTo_setup.py
```

✅ **Sortie attendue:** Configuration valide

### 3. Entraînement

```bash
# Avec sauvegarde Azure
python predictTo_train_model.py

# Sans Azure
python predictTo_train_model.py --no-azure
```

⏱️ **Durée:** 1-2 minutes

### 4. Test de Prédiction

```bash
python predictTo_predict_example.py
```

📊 **Résultat:** Prédiction du TO final avec analyse de tendance

---

## 📚 Documentation

| Fichier | Description | Niveau |
|---------|-------------|--------|
| **README.md** | Ce fichier - Vue d'ensemble et démarrage rapide | 🟢 Débutant |
| **GUIDE_COMPLET_PREDICTTO.md** | Guide utilisateur complet avec tous les cas d'usage | 🟡 Intermédiaire |
| **PREDICTTO_TRAINING_DOC.md** | Documentation technique pour comprendre et faire évoluer le système | 🔴 Avancé |

---

## 🎯 Ce que fait PredictTO

### Objectif
Prédire le **taux d'occupation final (TO à J+0)** en utilisant :
- 📊 Courbes de montée en charge (J-60 à J-7)
- 💰 Prix moyens compressés
- 🏷️ Numéro de cluster
- 📅 Features temporelles

### Performance
- 🎯 **Précision:** MAE = 5.6% (erreur moyenne)
- 📈 **R²:** 83% (variance expliquée)
- ⚡ **Rapidité:** Prédiction < 1ms

### Utilisation
```python
# Charger le modèle
from predictTo_predict_example import load_model_artifacts, predict_to
model, scaler, features = load_model_artifacts()

# Prédire
predicted_to = predict_to(
    model, scaler, features,
    to_series=[0.05, ..., 0.69],  # 53 valeurs
    pm_series=[120, ..., 125],     # 53 valeurs
    cluster=3, month=8, dayofweek=4
)

print(f"TO prédit: {predicted_to:.2%}")  # Ex: 72.34%
```

---

## ⚙️ Configuration

### Fichier YAML

`config_predictTo.yaml` permet de modifier :
- 📁 Chemins des données
- 🤖 Hyperparamètres du modèle
- ☁️ Configuration Azure
- 📊 Options de sortie

**Exemple de personnalisation:**
```yaml
model:
  n_estimators: 800      # Plus d'arbres
  learning_rate: 0.03    # Apprentissage plus lent
  max_depth: 9           # Arbres plus profonds
```

---

## ☁️ Azure Blob Storage

### Configuration

Le script supporte deux méthodes pour définir la chaîne de connexion Azure :

#### Méthode 1 : Fichier `.env` (Recommandé)

Créez un fichier `.env` à la racine du dossier `predictTo/` :

```bash
# .env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
```

✅ **Avantages :**
- Pas besoin de redéfinir la variable à chaque session
- Sécurisé (le fichier `.env` est ignoré par Git)
- Facile à partager avec l'équipe (via `.env.example`)

⚠️ **Important :** Ne commitez jamais le fichier `.env` dans Git (il contient des secrets).

#### Méthode 2 : Variable d'environnement système

```bash
# Windows (PowerShell)
$env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."

# Windows (CMD)
set AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...

# Linux/Mac
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
```

### Gestion des Modèles
```bash
# Lister les modèles disponibles
python load_predictTo_from_azure.py --list

# Télécharger le dernier modèle
python load_predictTo_from_azure.py --download latest
```

---

## 📊 Pipeline de Données

```
1. CHARGEMENT
   ├── clustering_results.csv (1415 observations)
   └── Indicateurs.csv (86k lignes)

2. PRÉPARATION
   ├── Pivot des PM par date
   ├── Calcul 7 features PM compressées
   └── Ajout features temporelles

3. MODÉLISATION
   ├── 64 features au total
   ├── Normalisation StandardScaler
   └── XGBoost (600 arbres)

4. ÉVALUATION
   ├── MAE, R², RMSE
   └── Graphiques de performance

5. SAUVEGARDE
   ├── Local: ../results/models/
   └── Azure: prediction-demande/models/
```

---

## 🛠️ Scripts Disponibles

| Script | Commande | Description |
|--------|----------|-------------|
| **Validation** | `python test_predictTo_setup.py` | Vérifie l'environnement |
| **Entraînement** | `python predictTo_train_model.py` | Entraîne le modèle |
| **Prédiction** | `python predictTo_predict_example.py` | Teste une prédiction |
| **Azure List** | `python load_predictTo_from_azure.py --list` | Liste les modèles |
| **Azure Download** | `python load_predictTo_from_azure.py --download latest` | Télécharge un modèle |

---

## 🐛 Résolution de Problèmes

### Erreur : "Module non trouvé"
```bash
pip install -r requirements_predictTo.txt --force-reinstall
```

### Erreur : "Fichier de données non trouvé"
```bash
# Vérifier les chemins (relatifs à predictTo/)
ls ../results/clustering_results.csv
ls ../data/Indicateurs.csv
```

### Erreur : "Azure connection failed"
```bash
# Vérifier la variable d'environnement
echo $env:AZURE_STORAGE_CONNECTION_STRING  # Windows PowerShell
echo %AZURE_STORAGE_CONNECTION_STRING%     # Windows CMD
echo $AZURE_STORAGE_CONNECTION_STRING      # Linux/Mac

# Vérifier si le fichier .env existe et est chargé
# Le script affichera un message au démarrage indiquant si .env a été détecté

# Ou désactiver Azure
python predictTo_train_model.py --no-azure
```

---

## 📞 Support

### Documentation
- 📖 **README.md** (ce fichier) - Vue d'ensemble et démarrage rapide
- 📚 **GUIDE_COMPLET_PREDICTTO.md** - Guide utilisateur complet avec tous les cas d'usage
- 📄 **PREDICTTO_TRAINING_DOC.md** - Documentation technique pour comprendre et faire évoluer le système

### Logs
- 📝 **predictTo_training.log** - Logs d'exécution détaillés

### Contact
Pour toute question, consultez la documentation ou les logs.

---

## ✅ Checklist de Démarrage

- [ ] Installation des dépendances (`pip install -r requirements_predictTo.txt`)
- [ ] Test de l'environnement (`python test_predictTo_setup.py`)
- [ ] Configuration Azure (optionnel)
- [ ] Premier entraînement (`python predictTo_train_model.py`)
- [ ] Vérification des résultats (logs + graphiques)
- [ ] Test de prédiction (`python predictTo_predict_example.py`)

---

## 🎉 Prêt !

Le système PredictTO est **prêt à l'emploi**.

**Commencez par:**
```bash
python test_predictTo_setup.py
```

Puis suivez les instructions affichées !

---

**Version:** 1.0  
**Date:** Décembre 2024  
**Licence:** Projet interne

