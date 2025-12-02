# Correction du bug horizon J-0

## 🐛 Problème identifié

Le script plantait lorsqu'on essayait d'entraîner un modèle avec `--horizon 0` (prédiction à J-0).

## 🔍 Cause du bug

### 1. **Data leakage** : Utilisation de données du jour J-0 dans les features

Le code original utilisait la condition `j_num >= horizon` pour filtrer les colonnes de features :

```python
# Code BUGGÉ (avant correction)
if j_num >= horizon:  # ❌ Pour horizon=0, cela incluait J-0 !
    pm_cols_available.append(col)
```

**Problème** : Pour `horizon=0`, cela incluait les colonnes `pm_J-0`, `ant_J-0`, `ds_J-0`, etc. dans les features, alors que **J-0 est le jour qu'on cherche à prédire** ! C'est du **data leakage**.

### Exemple concret

Pour prédire le TO du 15 janvier (J-0) :
- ❌ **Ancien code** : Utilisait `pm_J-0` (prix moyen du 15 janvier) → Data leakage !
- ✅ **Nouveau code** : Utilise uniquement `pm_J-1` à `pm_J-60` (données jusqu'au 14 janvier)

### 2. **Colonnes TO features vides**

Pour `horizon=0`, le range créait parfois une liste vide ou incluait J-0 :

```python
# Code BUGGÉ (avant correction)
to_feature_cols = [f"J-{i}" for i in range(60, horizon, -1)]
# Pour horizon=0: range(60, 0, -1) → [60, 59, ..., 1]  
# Cela exclut J-0 ✅ mais la logique était incohérente avec PM/Ant/Ds
```

## ✅ Solution apportée

### 1. **Changement de la condition de filtrage**

```python
# AVANT (BUGGÉ)
if j_num >= horizon:  # ❌ Incluait J-0 pour horizon=0
    pm_cols_available.append(col)

# APRÈS (CORRIGÉ)
if j_num > horizon:  # ✅ Exclut J-0 pour horizon=0
    pm_cols_available.append(col)
```

Cette correction a été appliquée pour :
- ✅ Features PM (Prix Moyen)
- ✅ Features Ant (Anticipation)
- ✅ Features Ds (Durée de Séjour)
- ✅ Features Comp (Prix Concurrents)
- ✅ Features TO (Taux d'Occupation historique)

### 2. **Ajout de validations**

Le code vérifie maintenant qu'il y a des features disponibles :

```python
if len(pm_cols_available) == 0:
    logger.error(f"❌ Aucune colonne PM disponible pour horizon={horizon}")
    raise ValueError(f"Pas de données PM disponibles pour horizon={horizon}")
```

### 3. **Messages de log améliorés**

```python
logger.info(f"Calcul des features PM sur données J-{horizon+1} à J-60 (pas de data leakage)")
logger.info(f"   Colonnes PM utilisées: {len(pm_cols_available)}")
```

## 📊 Comportement correct maintenant

### Pour horizon = 0 (J-0)

**Features utilisées** :
- TO historiques : `J-60, J-59, ..., J-2, J-1` (pas J-0 ❌)
- PM : `pm_J-1, pm_J-2, ..., pm_J-60` (pas pm_J-0 ❌)
- Ant : `ant_J-1, ant_J-2, ..., ant_J-60` (pas ant_J-0 ❌)
- Ds : `ds_J-1, ds_J-2, ..., ds_J-60` (pas ds_J-0 ❌)
- Comp : `CompPrixMedian_J-1, ..., CompPrixMedian_J-60` (pas J-0 ❌)

**Cible** : `J-0` (TO final du jour)

### Pour horizon = 7 (J-7)

**Features utilisées** :
- TO historiques : `J-60, J-59, ..., J-9, J-8` (pas J-7 à J-0 ❌)
- PM : `pm_J-8, pm_J-9, ..., pm_J-60` (pas pm_J-7 à pm_J-0 ❌)
- Ant : `ant_J-8, ant_J-9, ..., ant_J-60`
- Ds : `ds_J-8, ds_J-9, ..., ds_J-60`
- Comp : `CompPrixMedian_J-8, ..., CompPrixMedian_J-60`

**Cible** : `J-0` (TO final du jour)

## 🎯 Utilisation

Maintenant, l'entraînement avec horizon=0 fonctionne correctement :

```bash
# Entraînement pour J-0 uniquement
python predictTo_train_model.py --hotel D09 --horizon 0

# Batch training avec tous les horizons (incluant J-0)
python run_predictTo_batch.py --hotel D09
```

## 📝 Résumé de la logique

### Règle générale

Pour prédire à **J-0** (le jour du séjour), on ne peut utiliser que les données disponibles **AVANT** ce jour :

```
Horizon = H
├─ Features utilisables : J-(H+1) à J-60
├─ Features NON utilisables : J-H à J-0 (trop proches ou = cible)
└─ Cible à prédire : J-0
```

### Exemples

| Horizon | Features TO disponibles | Features PM/Ant/Ds disponibles | Cible |
|---------|------------------------|-------------------------------|-------|
| J-0 | J-1 à J-60 | pm_J-1 à pm_J-60 | J-0 |
| J-1 | J-2 à J-60 | pm_J-2 à pm_J-60 | J-0 |
| J-7 | J-8 à J-60 | pm_J-8 à pm_J-60 | J-0 |
| J-30 | J-31 à J-60 | pm_J-31 à pm_J-60 | J-0 |
| J-59 | J-60 | pm_J-60 | J-0 |

**Note** : 
- Pour J-59, on utilise uniquement les données de J-60 (1 seule colonne de features historiques)
- Pour J-60 et au-delà, il n'y aurait pas de données historiques disponibles (les données s'arrêtent à J-60)
- **L'horizon maximum supporté est donc J-59**

## ⚠️ Important

Cette correction **évite le data leakage** et garantit que :
1. Le modèle n'a accès qu'aux données disponibles au moment de la prédiction
2. Les performances rapportées sont réalistes
3. Le modèle peut être utilisé en production sans risque

## 🔗 Fichiers modifiés

- `predictTo/predictTo_train_model.py` : Corrections des filtres de colonnes
- `predictTo/docs/HORIZON_ZERO_FIX.md` : Ce document

