# Documentation des Features - Modèle XGBoost de Prédiction du TO

## Vue d'ensemble

Ce document décrit l'ensemble des features utilisées par le modèle XGBoost pour prédire le taux d'occupation (TO) à l'horizon J+7.

**Date de dernière mise à jour** : 2025-01-24

---

## Table des matières

1. [Features Taux d'Occupation Historique](#1-features-taux-doccupation-historique)
2. [Features Prix Moyen (PM) Compressées](#2-features-prix-moyen-pm-compressées)
3. [Features Anticipation (Ant) Compressées](#3-features-anticipation-ant-compressées)
4. [Features Durée de Séjour (Ds) Compressées](#4-features-durée-de-séjour-ds-compressées)
5. [Features Prix Concurrents Compressées](#5-features-prix-concurrents-compressées)
6. [Features Gap et Elasticity (PM vs Concurrents)](#6-features-gap-et-elasticity-pm-vs-concurrents)
7. [Features Temporelles](#7-features-temporelles)
8. [Features Jours Fériés et Vacances](#8-features-jours-fériés-et-vacances)
9. [Feature TO Année Précédente](#9-feature-to-année-précédente)
10. [Features Clustering](#10-features-clustering)
11. [Algorithmes de Compression](#11-algorithmes-de-compression)

---

## 1. Features Taux d'Occupation Historique

### Description
Série temporelle du taux d'occupation observé de J-60 à J-horizon.

### Source de données
Fichier `clustering_results.csv`

### Features
- **`J-60`, `J-59`, ..., `J-8`** (pour horizon=7)
- Nombre de features : **53** (de J-60 à J-8)

### Calcul
Valeurs directes du taux d'occupation à chaque point dans le temps.

### Principe anti-fuite (No Data Leakage)
⚠️ **Important** : Seules les données de **J-horizon à J-60** sont utilisées.
- Si horizon = 7, on utilise J-7 à J-60 (pas J-0 à J-6 qui seraient du futur)
- Cela garantit que le modèle ne voit pas de données futures lors de l'entraînement

---

## 2. Features Prix Moyen (PM) Compressées

### Description
Features calculées à partir de la série temporelle des prix moyens (PM) de l'hôtel.

### Source de données
Fichier `Indicateurs.csv` - Colonne `Pm`

### Features (7 au total)
1. **`pm_mean`** - Prix moyen global
2. **`pm_slope`** - Tendance (pente)
3. **`pm_volatility`** - Volatilité (écart-type)
4. **`pm_diff_sum`** - Somme des variations absolues
5. **`pm_change_ratio`** - Ratio de changement global
6. **`pm_last_jump`** - Variation récente
7. **`pm_trend_changes`** - Nombre de changements de direction

### Algorithme de calcul
Voir section [Algorithmes de Compression](#10-algorithmes-de-compression)

### Données utilisées
- Série PM de **J-horizon à J-60** (respect de l'horizon de prédiction)
- Conversion en numérique avec gestion des valeurs manquantes

---

## 3. Features Anticipation (Ant) Compressées

### Description
Features calculées à partir de la série temporelle de l'anticipation moyenne des réservations.

### Source de données
Fichier `Indicateurs.csv` - Colonne `Ant`

### Définition de l'anticipation
Nombre moyen de jours entre la date de réservation et la date de séjour.

### Features (6 au total)
1. **`ant_slope`** - Tendance de l'anticipation
2. **`ant_volatility`** - Volatilité de l'anticipation
3. **`ant_diff_sum`** - Somme des variations absolues
4. **`ant_change_ratio`** - Ratio de changement global
5. **`ant_last_jump`** - Variation récente
6. **`ant_trend_changes`** - Nombre de changements de direction

### Algorithme de calcul
Voir section [Algorithmes de Compression](#10-algorithmes-de-compression)

### Données utilisées
- Série Ant de **J-horizon à J-60**

### Interprétation
- **Slope positif** : L'anticipation augmente (les clients réservent de plus en plus tôt)
- **Volatility élevée** : Comportement de réservation instable
- **Last jump positif** : Augmentation récente de l'anticipation

---

## 4. Features Durée de Séjour (Ds) Compressées

### Description
Features calculées à partir de la série temporelle de la durée moyenne de séjour.

### Source de données
Fichier `Indicateurs.csv` - Colonne `Ds`

### Définition de la durée de séjour
Nombre moyen de nuits réservées par séjour.

### Features (6 au total)
1. **`ds_slope`** - Tendance de la durée de séjour
2. **`ds_volatility`** - Volatilité de la durée
3. **`ds_diff_sum`** - Somme des variations absolues
4. **`ds_change_ratio`** - Ratio de changement global
5. **`ds_last_jump`** - Variation récente
6. **`ds_trend_changes`** - Nombre de changements de direction

### Algorithme de calcul
Voir section [Algorithmes de Compression](#10-algorithmes-de-compression)

### Données utilisées
- Série Ds de **J-horizon à J-60**

### Interprétation
- **Slope positif** : Les séjours s'allongent
- **Volatility élevée** : Mix de séjours courts et longs
- **Last jump négatif** : Raccourcissement récent des séjours

---

## 5. Features Prix Concurrents Compressées

### Description
Features calculées à partir de la série temporelle du prix médian des concurrents.

### Source de données
Fichier `rateShopper.csv` - Colonne `CompPrixMedian`

### Features (6 au total)
1. **`comp_slope`** - Tendance des prix concurrents
2. **`comp_volatility`** - Volatilité des prix
3. **`comp_diff_sum`** - Somme des variations absolues
4. **`comp_change_ratio`** - Ratio de changement global
5. **`comp_last_jump`** - Variation récente
6. **`comp_trend_changes`** - Nombre de changements de direction

### Algorithme de calcul
Voir section [Algorithmes de Compression](#10-algorithmes-de-compression)

### Données utilisées
- Série CompPrixMedian de **J-horizon à J-60**
- Gestion des timezones (conversion en tz-naive)

### Interprétation
- **Slope positif** : Les concurrents augmentent leurs prix
- **Last jump élevé** : Mouvement récent important des concurrents

---

## 6. Features Gap et Elasticity (PM vs Concurrents)

### Description
Features calculées à partir de la comparaison entre vos prix (PM) et les prix médians des concurrents.

### Source de données
- **PM** : Fichier `Indicateurs.csv` - Colonne `Pm`
- **Concurrents** : Fichier `rateShopper.csv` - Colonne `CompPrixMedian`

### Séries construites

#### 6.1 Gap Series
```python
gap_series = PM_series - Comp_series
```
- **Description** : Différentiel de prix entre votre hôtel et le marché
- **Interprétation** :
  - > 0 : Votre prix est au-dessus du marché
  - < 0 : Votre prix est en-dessous du marché
  - = 0 : Prix aligné sur le marché

#### 6.2 Elasticity Series
```python
elasticity_series = PM_series / Comp_series
# Division par zéro gérée : Comp = 0 → NaN
```
- **Description** : Ratio de prix entre votre hôtel et le marché
- **Gestion** : Division par zéro remplacée par NaN
- **Interprétation** :
  - > 1 : Votre prix est plus élevé que le marché (ex: 1.2 = +20%)
  - < 1 : Votre prix est moins élevé que le marché (ex: 0.8 = -20%)
  - = 1 : Prix identique au marché

### Features extraites (4 au total)

#### 6.3 `gap_last`
- **Type** : Numérique
- **Description** : Différentiel PM - Comp au jour de l'horizon
- **Algorithme** :
```python
# gap_series construit de J-horizon à J-60
gap_last = gap_series[0]  # Première valeur = J-horizon
```
- **Utilité** : Indique votre positionnement prix actuel par rapport au marché à l'horizon de prédiction
- **Exemple** :
  - `gap_last = 20` → Vous êtes 20€ au-dessus du marché
  - `gap_last = -15` → Vous êtes 15€ en-dessous du marché

#### 6.4 `gap_slope`
- **Type** : Numérique
- **Description** : Tendance du différentiel de prix
- **Algorithme** :
```python
x = np.arange(len(gap_series_valid))
gap_slope = np.polyfit(x, gap_series_valid, 1)[0]
```
- **Interprétation** :
  - > 0 : Vous augmentez votre prix plus vite que le marché
  - < 0 : Le marché augmente plus vite que vous (ou vous baissez)
  - = 0 : Évolution parallèle au marché

#### 6.5 `elasticity_last`
- **Type** : Numérique
- **Description** : Ratio PM / Comp au jour de l'horizon
- **Algorithme** :
```python
# elasticity_series construit de J-horizon à J-60
elasticity_last = elasticity_series[0]  # Première valeur = J-horizon
# Si pas de données : elasticity_last = 1.0 (valeur neutre)
```
- **Utilité** : Mesure votre compétitivité prix actuelle à l'horizon de prédiction
- **Exemple** :
  - `elasticity_last = 1.15` → Vos prix sont 15% au-dessus du marché
  - `elasticity_last = 0.90` → Vos prix sont 10% en-dessous du marché

#### 6.6 `elasticity_slope`
- **Type** : Numérique
- **Description** : Tendance du ratio de prix
- **Algorithme** :
```python
x = np.arange(len(elasticity_series_valid))
elasticity_slope = np.polyfit(x, elasticity_series_valid, 1)[0]
```
- **Interprétation** :
  - > 0 : Votre ratio de prix augmente (vous devenez plus cher relativement)
  - < 0 : Votre ratio diminue (vous devenez plus compétitif)
  - = 0 : Ratio stable

### Données utilisées
- Séries PM et Comp de **J-horizon à J-60**
- Calcul point par point (gap et elasticity pour chaque jour)
- Seules les valeurs valides (non-NaN) sont utilisées pour les calculs

### Cas particuliers
- **Division par zéro** : Remplacée par NaN dans elasticity_series
- **Pas de données gap** : `gap_last = 0`, `gap_slope = 0`
- **Pas de données elasticity** : `elasticity_last = 1` (neutre), `elasticity_slope = 0`

### Utilité pour le modèle
- **Positionnement stratégique** : Capture votre politique de prix vs marché
- **Compétitivité** : Mesure si vous êtes premium, discount ou aligné
- **Dynamique** : Les slopes montrent si vous suivez ou devancez le marché
- **Prédictif** : La position prix influence directement le TO

---

## 7. Features Temporelles

### Description
Features extraites de la date de séjour pour capturer les patterns temporels.

### Features (2 au total)

#### 6.1 `month`
- **Description** : Mois de l'année de la date de séjour
- **Valeurs** : 1 à 12 (1=janvier, 12=décembre)
- **Source** : `stay_date.dt.month`
- **Utilité** : Capture la saisonnalité annuelle

#### 6.2 `dayofweek`
- **Description** : Jour de la semaine de la date de séjour
- **Valeurs** : 0 à 6 (0=lundi, 6=dimanche)
- **Source** : `stay_date.dt.dayofweek`
- **Utilité** : Capture les patterns semaine/weekend

---

## 7. Features Jours Fériés et Vacances

### Description
Features binaires ou numériques capturant les jours fériés français et périodes de vacances.

### Source de données
- Librairie `holidays` (jours fériés français)
- Librairie `vacances-scolaires-france` (vacances scolaires)

### Features (6 au total)

#### 7.1 `is_holiday_fr`
- **Type** : Binaire (0 ou 1)
- **Description** : 1 si la date de séjour est un jour férié français, 0 sinon
- **Algorithme** :
```python
is_holiday_fr = 1 if stay_date in holidays.France() else 0
```

#### 7.2 `is_bridge_day`
- **Type** : Binaire (0 ou 1)
- **Description** : 1 si la date est adjacente à un jour férié (pont), 0 sinon
- **Algorithme** :
```python
is_bridge_day = 1 if (
    (stay_date - 1 jour) est férié OR
    (stay_date + 1 jour) est férié
) else 0
```

#### 7.3 `days_to_holiday`
- **Type** : Numérique (0 à 90)
- **Description** : Nombre de jours jusqu'au prochain jour férié
- **Algorithme** :
```python
if stay_date est férié:
    days_to_holiday = 0
else:
    upcoming = [jours fériés futurs triés par date]
    if upcoming:
        days_to_holiday = min((upcoming[0] - stay_date).days, 90)
    else:
        days_to_holiday = 90
```

#### 7.4 `is_vacances_scolaires`
- **Type** : Binaire (0 ou 1)
- **Description** : 1 si la date est pendant les vacances scolaires (toutes zones A, B, C)
- **Algorithme** :
```python
is_vacances = 1 if (
    stay_date est en vacances zone A OR
    stay_date est en vacances zone B OR
    stay_date est en vacances zone C
) else 0
```

#### 7.5 `is_long_weekend_3j`
- **Type** : Binaire (0 ou 1)
- **Description** : 1 si la date fait partie d'un weekend de 3 jours
- **Configurations détectées** :
  - Vendredi-Samedi-Dimanche (vendredi férié)
  - Samedi-Dimanche-Lundi (lundi férié)

#### 7.6 `is_long_weekend_4j`
- **Type** : Binaire (0 ou 1)
- **Description** : 1 si la date fait partie d'un weekend de 4 jours
- **Configurations détectées** :
  - Jeudi-Vendredi-Samedi-Dimanche (jeudi ou vendredi férié)
  - Samedi-Dimanche-Lundi-Mardi (lundi ou mardi férié)

---

## 8. Feature TO Année Précédente

### Feature

#### `ToF1`
- **Type** : Numérique (0 à 1)
- **Description** : Taux d'occupation final (J-0) de la même date l'année précédente
- **Algorithme** :
```python
# Pour chaque stay_date :
stay_date_last_year = stay_date - 1 an
ToF1 = J-0 de stay_date_last_year si disponible, sinon 0
```

### Utilité
- Capture la saisonnalité annuelle
- Permet de comparer avec l'année précédente
- Détecte les tendances à long terme

### Exemple
- Si TO du 25 décembre 2023 = 0.85
- Alors ToF1 pour le 25 décembre 2024 = 0.85

---

## 9. Features Clustering

### Description
Features issues de l'analyse de clustering préalable.

### Features (2 au total)

#### 9.1 `nb_observations`
- **Type** : Numérique (entier)
- **Description** : Nombre de points de données **réellement utilisables** dans la série temporelle du TO en fonction de l'horizon
- **Source initiale** : Fichier `clustering_results.csv`
- **Recalcul** : ⚠️ **IMPORTANT** - Cette feature est recalculée lors de la préparation des données pour refléter l'horizon de prédiction
- **Algorithme de recalcul** :
```python
# Pour horizon = 7 :
# Compter les valeurs non-NaN de J-60 à J-7 (pas J-0 !)
to_cols_available = ["J-60", "J-59", ..., "J-8", "J-7"]
nb_observations = nombre de valeurs non-NaN dans to_cols_available
```
- **Valeurs selon l'horizon** : 
  - **horizon = 7** : max 54 observations (J-60 à J-7)
  - **horizon = 14** : max 47 observations (J-60 à J-14)
  - **horizon = 30** : max 31 observations (J-60 à J-30)
- **Utilité** : 
  - Indicateur de complétude des données **réellement disponibles au moment de la prédiction**
  - Permet au modèle d'ajuster sa confiance selon la quantité de données utilisables
  - Détecte les nouveaux hôtels ou périodes avec données manquantes
  - **Cohérent avec le principe anti-fuite** : ne compte que les données accessibles
- **Exemple avec horizon = 7** : 
  - `nb_observations = 54` → Historique complet de J-60 à J-7 (série complète)
  - `nb_observations = 30` → Seulement 30 jours d'historique disponibles (hôtel récent ou données partielles)
  - `nb_observations = 10` → Très peu de données historiques (nouvel hôtel)

#### 9.2 `cluster`
- **Type** : Catégorielle (entier)
- **Description** : Identifiant du cluster auquel appartient le pattern TO
- **Source** : Fichier `clustering_results.csv`
- **Utilité** : Regroupe les dates avec des comportements similaires

---

## 10. Algorithmes de Compression

### Description
Les séries temporelles (PM, Ant, Ds, Concurrents) sont compressées en 6-7 features résumant leur comportement.

### Fonction : `compute_price_features(price_series, prefix)`

#### Étape 1 : Nettoyage des données
```python
# Conversion en numérique
series = pd.to_numeric(series, errors='coerce')

# Remplacement des infinis
series = series.replace([np.inf, -np.inf], np.nan)

# Suppression des valeurs manquantes
valid_series = series.dropna()
```

#### Étape 2 : Calcul des features

##### 2.1 `{prefix}_slope` - Tendance (pente)
```python
x = np.arange(len(valid_series))
slope = np.polyfit(x, valid_series, 1)[0]
```
- **Interprétation** :
  - > 0 : Tendance à la hausse
  - < 0 : Tendance à la baisse
  - = 0 : Stable

##### 2.2 `{prefix}_volatility` - Volatilité
```python
volatility = valid_series.std()
```
- **Interprétation** :
  - Élevée : Valeurs très variables
  - Faible : Valeurs stables

##### 2.3 `{prefix}_diff_sum` - Somme des variations absolues
```python
diff_sum = np.sum(np.abs(np.diff(valid_series)))
```
- **Interprétation** : Mesure l'amplitude totale des variations

##### 2.4 `{prefix}_change_ratio` - Ratio de changement global
```python
first_value = valid_series[0]
last_value = valid_series[-1]
change_ratio = (last_value - first_value) / first_value if first_value != 0 else 0
```
- **Interprétation** :
  - > 0 : Augmentation globale
  - < 0 : Diminution globale

##### 2.5 `{prefix}_last_jump` - Variation récente
```python
if len(valid_series) >= 6:
    last_jump = last_value - valid_series[-6]
else:
    last_jump = last_value - first_value
```
- **Interprétation** : Capture les mouvements récents (derniers 6 points)

##### 2.6 `{prefix}_trend_changes` - Changements de direction
```python
diffs = np.diff(valid_series)
signs = np.sign(diffs)
trend_changes = int(np.sum(np.diff(signs) != 0))
```
- **Interprétation** :
  - Élevé : Comportement erratique
  - Faible : Tendance stable

#### Cas particuliers

**Si toutes les valeurs sont NaN** :
```python
Toutes les features = 0
```

**Si moins de 2 points valides** :
```python
slope = 0
volatility = 0
diff_sum = 0
change_ratio = 0
last_jump = 0
trend_changes = 0
```

---

## Récapitulatif des Features

### Nombre total de features

| Catégorie | Nombre | Détail |
|-----------|--------|--------|
| TO historiques | 53 | J-60 à J-8 (pour horizon=7) |
| PM compressées | 7 | mean + 6 features |
| Ant compressées | 6 | 6 features |
| Ds compressées | 6 | 6 features |
| Concurrents compressés | 6 | 6 features |
| Gap/Elasticity | 4 | gap_last, gap_slope, elasticity_last, elasticity_slope |
| Temporelles | 2 | month, dayofweek |
| Jours fériés/vacances | 6 | is_holiday_fr, is_bridge_day, days_to_holiday, is_vacances_scolaires, is_long_weekend_3j, is_long_weekend_4j |
| TO année précédente | 1 | ToF1 |
| Clustering | 2 | nb_observations, cluster |
| **TOTAL** | **93** | **(pour horizon=7)** |

---

## Normalisation

Toutes les features sont normalisées avec `StandardScaler` avant d'être passées au modèle XGBoost :

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

- **Moyenne** : 0
- **Écart-type** : 1

---

## Principe anti-fuite (No Data Leakage)

⚠️ **Point critique** : Pour toutes les séries temporelles (PM, Ant, Ds, Concurrents), seules les données de **J-horizon à J-60** sont utilisées.

### Exemple avec horizon = 7
- ✅ **Utilisé** : J-7, J-8, J-9, ..., J-60
- ❌ **NON utilisé** : J-0, J-1, J-2, J-3, J-4, J-5, J-6 (ce serait du futur !)

Cela garantit que le modèle simule une situation réelle où on ne connaît pas les valeurs futures.

---

## Fichiers sources

| Fichier | Colonnes utilisées | Description |
|---------|-------------------|-------------|
| `clustering_results.csv` | stay_date, hotCode, J-60 à J-0, nb_observations, cluster | TO historique et clustering |
| `Indicateurs.csv` | Date, ObsDate, Pm, Ant, Ds | Prix moyen, anticipation, durée séjour |
| `rateShopper.csv` | Date, DateImport, CompPrixMedian | Prix concurrents |

---

## Notes techniques

### Gestion des valeurs manquantes
- Les valeurs manquantes dans les séries temporelles sont ignorées lors du calcul des features compressées
- Si toute une série est manquante, les features valent 0
- Les lignes avec des NaN dans les features finales sont supprimées avant l'entraînement

### Performance
- Le calcul des features compressées se fait ligne par ligne (itération sur DataFrame)
- Pour améliorer la performance, utiliser `.apply()` ou vectorisation quand possible

---

## Tableau récapitulatif de toutes les Features

### Liste complète des features par catégorie

| # | Feature | Type | Description | Algorithme/Source |
|---|---------|------|-------------|-------------------|
| **1. TO HISTORIQUES (53 features pour horizon=7)** |
| 1-53 | `J-60` à `J-8` | Numérique (0-1) | Taux d'occupation historique de J-60 à J-horizon | Valeurs directes des TO observés |
| **2. FEATURES PM - PRIX MOYEN (7 features)** |
| 54 | `pm_mean` | Numérique | Prix moyen global de la série PM | `mean(PM_series)` |
| 55 | `pm_slope` | Numérique | Tendance des prix moyens | Régression linéaire : `polyfit(x, PM, 1)[0]` |
| 56 | `pm_volatility` | Numérique | Volatilité des prix moyens | Écart-type : `std(PM_series)` |
| 57 | `pm_diff_sum` | Numérique | Somme des variations absolues de PM | `sum(abs(diff(PM_series)))` |
| 58 | `pm_change_ratio` | Numérique | Ratio de changement global de PM | `(last - first) / first` |
| 59 | `pm_last_jump` | Numérique | Variation récente de PM (6 derniers points) | `last - PM[-6]` |
| 60 | `pm_trend_changes` | Entier | Nombre de changements de direction | Comptage des changements de signe dans les différences |
| **3. FEATURES ANT - ANTICIPATION (6 features)** |
| 61 | `ant_slope` | Numérique | Tendance de l'anticipation moyenne | Régression linéaire sur série Ant |
| 62 | `ant_volatility` | Numérique | Volatilité de l'anticipation | Écart-type de la série Ant |
| 63 | `ant_diff_sum` | Numérique | Somme des variations absolues d'anticipation | `sum(abs(diff(Ant_series)))` |
| 64 | `ant_change_ratio` | Numérique | Ratio de changement de l'anticipation | `(last - first) / first` |
| 65 | `ant_last_jump` | Numérique | Variation récente d'anticipation | `last - Ant[-6]` |
| 66 | `ant_trend_changes` | Entier | Changements de direction dans l'anticipation | Comptage des changements de signe |
| **4. FEATURES DS - DURÉE DE SÉJOUR (6 features)** |
| 67 | `ds_slope` | Numérique | Tendance de la durée moyenne de séjour | Régression linéaire sur série Ds |
| 68 | `ds_volatility` | Numérique | Volatilité de la durée de séjour | Écart-type de la série Ds |
| 69 | `ds_diff_sum` | Numérique | Somme des variations absolues de durée | `sum(abs(diff(Ds_series)))` |
| 70 | `ds_change_ratio` | Numérique | Ratio de changement de durée | `(last - first) / first` |
| 71 | `ds_last_jump` | Numérique | Variation récente de durée | `last - Ds[-6]` |
| 72 | `ds_trend_changes` | Entier | Changements de direction dans la durée | Comptage des changements de signe |
| **5. FEATURES CONCURRENTS - PRIX (6 features)** |
| 73 | `comp_slope` | Numérique | Tendance des prix médians concurrents | Régression linéaire sur CompPrixMedian |
| 74 | `comp_volatility` | Numérique | Volatilité des prix concurrents | Écart-type de CompPrixMedian |
| 75 | `comp_diff_sum` | Numérique | Somme des variations de prix concurrents | `sum(abs(diff(Comp_series)))` |
| 76 | `comp_change_ratio` | Numérique | Ratio de changement des prix concurrents | `(last - first) / first` |
| 77 | `comp_last_jump` | Numérique | Variation récente des prix concurrents | `last - Comp[-6]` |
| 78 | `comp_trend_changes` | Entier | Changements de direction des prix | Comptage des changements de signe |
| **6. FEATURES GAP ET ELASTICITY - PM vs CONCURRENTS (4 features)** |
| 79 | `gap_last` | Numérique | Différentiel PM - Comp au jour horizon | `gap_series[0]` où gap = PM - Comp (première valeur = J-horizon) |
| 80 | `gap_slope` | Numérique | Tendance du différentiel PM - Comp | Régression linéaire sur `gap_series` (J-horizon à J-60) |
| 81 | `elasticity_last` | Numérique | Ratio PM / Comp au jour horizon | `elasticity_series[0]` où elasticity = PM / Comp (première valeur = J-horizon) |
| 82 | `elasticity_slope` | Numérique | Tendance du ratio PM / Comp | Régression linéaire sur `elasticity_series` (J-horizon à J-60) |
| **7. FEATURES TEMPORELLES (2 features)** |
| 83 | `month` | Entier (1-12) | Mois de l'année de la date de séjour | `stay_date.dt.month` |
| 84 | `dayofweek` | Entier (0-6) | Jour de la semaine (0=lundi, 6=dimanche) | `stay_date.dt.dayofweek` |
| **8. FEATURES JOURS FÉRIÉS ET VACANCES (6 features)** |
| 85 | `is_holiday_fr` | Binaire (0/1) | 1 si jour férié français, 0 sinon | `holidays.France()` |
| 86 | `is_bridge_day` | Binaire (0/1) | 1 si date adjacente à un férié (pont) | Vérification J-1 ou J+1 = férié |
| 87 | `days_to_holiday` | Entier (0-90) | Nombre de jours jusqu'au prochain férié | Distance au prochain férié, max=90 |
| 88 | `is_vacances_scolaires` | Binaire (0/1) | 1 si vacances scolaires (zones A/B/C) | `vacances_scolaires_france` |
| 89 | `is_long_weekend_3j` | Binaire (0/1) | 1 si weekend de 3 jours | Détection Ven-Sam-Dim ou Sam-Dim-Lun |
| 90 | `is_long_weekend_4j` | Binaire (0/1) | 1 si weekend de 4 jours | Détection Jeu-Ven-Sam-Dim ou Sam-Dim-Lun-Mar |
| **9. FEATURE TO ANNÉE PRÉCÉDENTE (1 feature)** |
| 91 | `ToF1` | Numérique (0-1) | TO final (J-0) de la même date année N-1 | Self-join avec `stay_date - 1 an` |
| **10. FEATURES CLUSTERING (2 features)** |
| 92 | `nb_observations` | Entier | Nombre de points TO disponibles (J-60 à J-horizon) | Comptage des valeurs non-NaN recalculé selon horizon |
| 93 | `cluster` | Entier | Identifiant du cluster de patterns similaires | Résultat du clustering préalable |

### Légende des types

- **Numérique** : Valeur continue (float)
- **Entier** : Valeur discrète (int)
- **Binaire** : 0 ou 1
- **0-1** : Valeur normalisée entre 0 et 1 (taux)

### Notes importantes

1. **Normalisation** : Toutes les features sont normalisées avec `StandardScaler` avant l'entraînement
2. **Horizon** : Le nombre de features TO historiques dépend de l'horizon de prédiction
   - horizon=7 → 53 features (J-60 à J-8, excluant J-7 à J-0)
   - horizon=14 → 46 features (J-60 à J-15, excluant J-14 à J-0)
3. **Data Leakage** : Toutes les séries temporelles (PM, Ant, Ds, Concurrents, Gap, Elasticity) sont calculées uniquement sur J-horizon à J-60
4. **Gap/Elasticity** : Calculées à partir des séries PM et Concurrents (CompPrixMedian)
   - `gap_series = PM - Comp` (écart de prix)
   - `elasticity_series = PM / Comp` (ratio de prix, division par 0 gérée)
5. **Total** : **93 features** pour horizon=7

---

**Auteur** : Équipe Data Science  
**Version** : 1.0  
**Date** : 2025-01-24

