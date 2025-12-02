# Guide de débogage du R²

## 🔍 Problème : R² anormal (> 1, très négatif, ou valeur bizarre)

Si vous observez un R² avec une valeur anormale (par exemple 57 au lieu de 0.57), voici les étapes pour diagnostiquer et résoudre le problème.

## ✅ Ce qu'est un R² normal

Le **coefficient de détermination (R²)** mesure la qualité de la prédiction :
- **R² = 1.0** : Prédictions parfaites
- **R² = 0.85-0.95** : Très bonnes prédictions (typique pour un bon modèle)
- **R² = 0.7-0.85** : Bonnes prédictions
- **R² = 0.5-0.7** : Prédictions correctes
- **R² = 0.0** : Prédictions aussi bonnes qu'une simple moyenne
- **R² < 0** : Prédictions pires qu'une simple moyenne (modèle problématique)

**Important** : Le R² est TOUJOURS entre -∞ et 1.0 (en pratique, rarement en dessous de -10)

## 🐛 Causes possibles d'un R² anormal

### 1. **Problème d'extraction dans le script batch**

**Symptôme** : Vous voyez un R² de 57, 30, ou une autre valeur étrange dans le résumé du batch.

**Cause** : Le script batch extrait incorrectement la valeur depuis les logs.

**Solution** : 
- ✅ **Corrigé** dans la dernière version du code
- Le problème venait du parsing de la ligne de log qui contenait l'horodatage
- Exemple : `2025-01-15 10:30:45 - INFO - Test R²: 0.8567`
- L'ancien code prenait la valeur après le premier `:` → `30` au lieu de `0.8567`

### 2. **Fuite de données (Data Leakage)**

**Symptôme** : R² > 0.99 (trop parfait) ou même > 1.0

**Cause** : Le modèle a accès à des informations futures qu'il ne devrait pas avoir.

**Vérifications** :
```python
# Dans le code, vérifiez que :
# 1. Les features PM/Ant/Ds utilisent bien J-{horizon} à J-60
horizon = 7
pm_cols_available = [f"pm_J-{i}" for i in range(horizon, 61)]

# 2. Pas de colonnes TO futures dans les features
# Par exemple, pour horizon=7, ne pas utiliser J-0, J-1, ..., J-6
```

**Solution** : Vérifier le code de préparation des features dans `prepare_data()`

### 3. **Données corrompues ou NaN/Inf**

**Symptôme** : R² bizarre, erreurs de calcul

**Cause** : Présence de valeurs NaN, Inf, ou données corrompues

**Diagnostic** :
```python
# Vérifier dans le code ou ajouter des prints
print(f"NaN dans X: {X.isna().sum().sum()}")
print(f"NaN dans y: {y.isna().sum()}")
print(f"Inf dans X: {np.isinf(X).sum().sum()}")
```

**Solution** : Le code filtre déjà les NaN, mais vérifiez vos données sources

### 4. **Problème de normalisation**

**Symptôme** : R² négatif ou très faible

**Cause** : Normalisation appliquée sur le mauvais ensemble ou double normalisation

**Vérification** :
```python
# Le code devrait faire (dans train_model):
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, ...)
X_train = scaler.fit_transform(X_train_raw)  # Fit sur train uniquement
X_test = scaler.transform(X_test_raw)        # Transform sur test
```

### 5. **Erreur d'affichage / formatage**

**Symptôme** : Le calcul est correct mais l'affichage montre une valeur bizarre

**Diagnostic** : Regardez les logs détaillés :
```bash
# Dans les logs, cherchez :
📊 Test R²: 0.8567  # ← Valeur correcte
```

**Test** : Exécutez le script de test :
```bash
cd predictTo
python test_r2_calculation.py
```

## 🔧 Corrections apportées

### Version corrigée (actuelle)

1. **Extraction des métriques améliorée** (`run_predictTo_batch.py`) :
   ```python
   # Avant (incorrect)
   r2_value = line.split(':')[1].strip()  # ❌ Prend le mauvais segment
   
   # Après (correct)
   r2_value = line.split('Test R²:')[1].strip()  # ✅ Extrait après "Test R²:"
   ```

2. **Validation du R²** ajoutée dans `predictTo_train_model.py` :
   ```python
   if test_r2 < -1 or test_r2 > 1.1:
       logger.warning(f"⚠️  ATTENTION: R² test anormal ({test_r2:.6f})")
   ```

3. **Affichage amélioré** :
   ```python
   logger.info(f"   Test R²:    {results['test']['r2']:.4f}")
   # Affiche toujours 4 décimales (ex: 0.8567)
   ```

## 📝 Comment déboguer étape par étape

### Étape 1 : Tester le calcul du R²
```bash
cd predictTo
python test_r2_calculation.py
```
✅ Si ce test passe → Le calcul sklearn fonctionne correctement

### Étape 2 : Regarder les logs détaillés
```bash
# Après un entraînement, ouvrir le fichier de log
cat predictTo_training.log | grep "R²"
```
Cherchez la ligne :
```
📊 Test R²: 0.XXXX
```

### Étape 3 : Comparer avec le résumé batch
Si vous utilisez `run_predictTo_batch.py`, comparez :
- **Dans `predictTo_training.log`** : R² individuel
- **Dans `predictTo_batch.log`** : R² du résumé

Ils devraient être identiques.

### Étape 4 : Vérifier les prédictions de test
```bash
# Ouvrir le fichier CSV des prédictions
# results/D09/{hotel}/J-{horizon}/test_predictions.csv

# Calculer manuellement le R² avec Python
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv('results/D09/D09/J-7/test_predictions.csv', sep=';')
r2_manual = r2_score(df['y_test'], df['y_pred'])
print(f"R² calculé manuellement : {r2_manual:.4f}")
```

### Étape 5 : Vérifier les données sources
```python
# Charger et inspecter les données
import pandas as pd

clusters = pd.read_csv('../cluster/results/D09/clustering_results.csv', sep=';')
print(f"Clusters shape: {clusters.shape}")
print(f"Colonnes TO disponibles: {[c for c in clusters.columns if c.startswith('J-')]}")
print(f"NaN dans TO: {clusters[[c for c in clusters.columns if c.startswith('J-')]].isna().sum().sum()}")
```

## 🎯 Valeurs attendues selon l'horizon

Le R² varie généralement selon l'horizon de prédiction :

| Horizon | R² attendu | Explication |
|---------|------------|-------------|
| J-0     | > 0.95     | Prédiction le jour même (très facile) |
| J-1     | 0.90-0.95  | Prédiction à 1 jour (facile) |
| J-3     | 0.85-0.92  | Prédiction à 3 jours (bon) |
| J-7     | 0.80-0.90  | Prédiction à 7 jours (correct) |
| J-14    | 0.75-0.85  | Prédiction à 14 jours (acceptable) |
| J-30    | 0.65-0.80  | Prédiction à 30 jours (difficile) |
| J-60    | 0.55-0.75  | Prédiction à 60 jours (très difficile) |

**Important** : Ces valeurs sont indicatives. Un R² plus faible peut être normal si :
- L'hôtel a une forte variabilité
- Peu de données historiques
- Événements imprévisibles (congrès, travaux, etc.)

## 📞 Support

Si le problème persiste après toutes ces vérifications :

1. **Créer un rapport** avec :
   - Le R² affiché (bizarre)
   - Les logs complets (`predictTo_training.log`)
   - La commande exécutée
   - Les 5 premières lignes de `test_predictions.csv`

2. **Vérifier la version** :
   ```bash
   git log --oneline -1
   ```

3. **Réentraîner avec logs détaillés** :
   ```bash
   python predictTo_train_model.py --hotel D09 --horizon 7 --no-azure 2>&1 | tee debug.log
   ```

