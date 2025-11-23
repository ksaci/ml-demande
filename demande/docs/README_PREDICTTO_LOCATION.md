# 📍 PredictTO - Nouvelle Localisation

## ⚠️ Important : Les fichiers ont été déplacés !

Tous les fichiers du système **PredictTO** sont maintenant dans le sous-dossier :

```
📂 demande/predictTo/
```

---

## 🚀 Pour Utiliser PredictTO

### Étape 1 : Se Déplacer dans le Bon Dossier

```bash
cd demande/predictTo
```

### Étape 2 : Suivre la Documentation

Ouvrez **[predictTo/README.md](predictTo/README.md)** pour commencer.

Ou lancez directement :

```bash
cd demande/predictTo
python test_predictTo_setup.py
```

---

## 📂 Structure Complète

```
demande/
├── README_PREDICTTO_LOCATION.md    ← VOUS ÊTES ICI (fichier indicateur)
│
├── predictTo/                      ← ALLEZ ICI pour PredictTO
│   ├── README.md                   # Commencez par ce fichier
│   ├── INDEX.md                    # Navigation documentation
│   ├── DEMARRAGE_RAPIDE.md         # 3 étapes pour commencer
│   │
│   ├── predictTo_train_model.py    # Scripts Python
│   ├── predictTo_predict_example.py
│   ├── test_predictTo_setup.py
│   ├── load_predictTo_from_azure.py
│   │
│   ├── config_predictTo.yaml       # Configuration
│   ├── requirements_predictTo.txt  # Dépendances
│   │
│   └── *.md                        # Toute la documentation
│
├── data/
│   └── Indicateurs.csv             # Données utilisées par PredictTO
│
└── results/
    ├── clustering_results.csv      # Résultats utilisés par PredictTO
    └── models/                     # Modèles générés par PredictTO
```

---

## 📖 Liens Rapides

### Documentation Principale

👉 **[predictTo/README.md](predictTo/README.md)** - Commencez ici !

### Démarrage Rapide

👉 **[predictTo/DEMARRAGE_RAPIDE.md](predictTo/DEMARRAGE_RAPIDE.md)** - 3 étapes

### Navigation Complète

👉 **[predictTo/INDEX.md](predictTo/INDEX.md)** - Toute la documentation

---

## 🎯 Commandes Rapides

```bash
# Se déplacer dans le dossier
cd demande/predictTo

# Installer
pip install -r requirements_predictTo.txt

# Tester
python test_predictTo_setup.py

# Entraîner
python predictTo_train_model.py

# Prédire
python predictTo_predict_example.py
```

---

## ⚡ Pourquoi ce Changement ?

### Avantages

✅ **Organisation** - Projet isolé dans son dossier  
✅ **Clarté** - Nommage cohérent (predictTo partout)  
✅ **Maintenabilité** - Plus facile à gérer  
✅ **Déploiement** - Facile à packager  
✅ **Documentation** - Centralisée au même endroit  

---

## 📞 Pour Toute Question

Consultez la documentation dans **[demande/predictTo/](predictTo/)**

**Fichier le plus utile pour commencer:**  
👉 **[predictTo/README.md](predictTo/README.md)**

---

**Date de migration:** 16 Décembre 2024  
**Nouveau dossier:** `demande/predictTo/`  
**Statut:** ✅ Opérationnel

