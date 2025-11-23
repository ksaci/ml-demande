# 📍 PredictTO - Système de Prédiction du Taux d'Occupation

## ⚠️ Les Fichiers Ont Été Déplacés !

**Nouvelle localisation:** `demande/predictTo/`

---

## 🚀 Pour Utiliser PredictTO

### Étape 1 : Accédez au Bon Dossier

```bash
cd demande/predictTo
```

### Étape 2 : Commencez par la Documentation

👉 **Ouvrez : [predictTo/START_HERE.md](predictTo/START_HERE.md)**

---

## 📂 Structure Complète

```
demande/
│
├── README_PREDICTTO.md                 ← VOUS ÊTES ICI (pointeur)
│
├── predictTo/                          ← TOUT EST LÀ-BAS !
│   ├── START_HERE.md                   ← Commencez par ce fichier
│   ├── INDEX.md                        ← Navigation documentation
│   ├── README.md                       ← Vue d'ensemble
│   ├── DEMARRAGE_RAPIDE.md             ← 3 étapes rapides
│   │
│   ├── predictTo_train_model.py        ← Scripts Python
│   ├── predictTo_predict_example.py
│   ├── test_predictTo_setup.py
│   ├── load_predictTo_from_azure.py
│   │
│   ├── config_predictTo.yaml           ← Configuration
│   ├── requirements_predictTo.txt      ← Dépendances
│   │
│   └── 6 autres fichiers .md           ← Documentation complète
│
├── data/
│   └── Indicateurs.csv                 ← Données PM (utilisées par PredictTO)
│
└── results/
    ├── clustering_results.csv          ← Résultats clustering
    └── models/                         ← Modèles générés
```

---

## 🎯 Liens Rapides

### Documentation Principale

👉 **[predictTo/START_HERE.md](predictTo/START_HERE.md)** - COMMENCEZ ICI !  
👉 **[predictTo/INDEX.md](predictTo/INDEX.md)** - Navigation complète  
👉 **[predictTo/README.md](predictTo/README.md)** - Vue d'ensemble  

---

## ⚡ Démarrage Ultra-Rapide

```bash
# 1. Accédez au dossier
cd demande/predictTo

# 2. Installez
pip install -r requirements_predictTo.txt

# 3. Testez
python test_predictTo_setup.py

# 4. Lancez
python predictTo_train_model.py --no-azure
```

---

## 📞 Support

**Toute la documentation est dans :** `demande/predictTo/`

**Commencez par :** [predictTo/START_HERE.md](predictTo/START_HERE.md)

---

**📂 Nouvelle localisation:** `demande/predictTo/`  
**📅 Date de migration:** 16 Décembre 2024  
**✅ Statut:** Opérationnel
