# Guide de débogage des erreurs d'entraînement

## 🐛 Gestion améliorée des erreurs

Le script batch `run_predictTo_batch.py` a été amélioré pour afficher des messages d'erreur détaillés au lieu de simples codes retour.

## 📋 Que faire en cas d'erreur ?

### 1. **Consulter les logs dans le terminal**

Lorsqu'une erreur se produit, le script affiche maintenant :

```
❌ Erreur lors de l'entraînement J-7
Code retour: 1

================================================================================
DÉTAILS DE L'ERREUR (STDERR):
================================================================================
FileNotFoundError: [Errno 2] No such file or directory: '../cluster/results/6N8/clustering_results.csv'

================================================================================
SORTIE STANDARD (STDOUT - dernières 50 lignes):
================================================================================
2025-01-15 10:30:45 - INFO - Chargement des données...
2025-01-15 10:30:45 - INFO - Chargement des clusters depuis: ../cluster/results/6N8/clustering_results.csv
2025-01-15 10:30:45 - ERROR - ❌ Erreur lors du chargement des données: [Errno 2] No such file or directory
Traceback (most recent call last):
  File "predictTo_train_model.py", line 169, in load_data
    clusters = pd.read_csv(clustering_path, sep=';')
FileNotFoundError: [Errno 2] No such file or directory: '../cluster/results/6N8/clustering_results.csv'
================================================================================
```

### 2. **Consulter les fichiers de logs détaillés**

Les logs complets sont automatiquement sauvegardés dans le dossier `error_logs/` :

```
predictTo/
└── error_logs/
    ├── error_6N8_J-7_20250115_103045.log
    ├── error_6N8_J-14_20250115_103145.log
    └── error_D09_J-30_20250115_104500.log
```

Chaque fichier contient :
- Le code retour
- L'erreur complète (STDERR)
- La sortie complète (STDOUT) avec tous les logs

### 3. **Consulter le résumé final**

À la fin du batch, les erreurs sont résumées de façon structurée :

```
⚠️  ERREURS DÉTAILLÉES:
================================================================================

🔴 Horizon J-7 (error):
--------------------------------------------------------------------------------
   Code retour: 1
   STDERR:
   FileNotFoundError: [Errno 2] No such file or directory: '../cluster/results/6N8/clustering_results.csv'
   
   STDOUT (dernières lignes):
   2025-01-15 10:30:45 - INFO - Chargement des données...
   2025-01-15 10:30:45 - ERROR - ❌ Erreur lors du chargement des données
   Traceback (most recent call last):
     ...
--------------------------------------------------------------------------------
================================================================================
```

## 🔍 Erreurs courantes et solutions

### 1. **FileNotFoundError: clustering_results.csv**

**Erreur** :
```
FileNotFoundError: [Errno 2] No such file or directory: '../cluster/results/6N8/clustering_results.csv'
```

**Cause** : Les résultats de clustering n'existent pas pour cet hôtel.

**Solution** :
```bash
# Exécuter d'abord le clustering pour cet hôtel
cd cluster
python prediction_cluster.py --hotel 6N8

# Puis relancer l'entraînement
cd ../predictTo
python run_predictTo_batch.py --hotel 6N8
```

### 2. **FileNotFoundError: Indicateurs.csv**

**Erreur** :
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/6N8/Indicateurs.csv'
```

**Cause** : Les données d'indicateurs n'existent pas pour cet hôtel.

**Solution** :
- Vérifier que le fichier existe dans `data/{hotel}/Indicateurs.csv`
- Copier les données depuis la source si nécessaire

### 3. **ValueError: Aucune colonne PM disponible**

**Erreur** :
```
ValueError: Pas de données PM disponibles pour horizon=7
```

**Cause** : Les colonnes PM nécessaires n'existent pas dans les données.

**Solution** :
- Vérifier que `Indicateurs.csv` contient bien les colonnes `Pm` pour J-7 à J-60
- Vérifier le format du fichier (séparateur `;`)

### 4. **ValueError: Aucune donnée pour l'hôtel**

**Erreur** :
```
ValueError: Aucune donnée trouvée pour l'hôtel 6N8
```

**Cause** : Le filtrage par `hotCode` ne retourne aucune ligne.

**Solution** :
- Vérifier que `hotCode` est bien `6N8` dans les fichiers CSV
- Vérifier qu'il n'y a pas d'espace ou de caractère bizarre dans les données

### 5. **MemoryError ou ressources insuffisantes**

**Erreur** :
```
MemoryError: Unable to allocate array
```

**Cause** : Pas assez de RAM disponible.

**Solution** :
- Fermer d'autres applications
- Réduire le nombre d'horizons à entraîner simultanément
- Utiliser `--horizons` pour entraîner par petits lots :
  ```bash
  python run_predictTo_batch.py --hotel 6N8 --horizons 7 14 30
  python run_predictTo_batch.py --hotel 6N8 --horizons 1 3 5
  ```

## 📊 Logs disponibles

### 1. **predictTo_batch.log**
Logs du script batch principal (résumé)

### 2. **predictTo_training.log**
Logs détaillés de chaque entraînement

### 3. **error_logs/error_{hotel}_J-{horizon}_{timestamp}.log**
Logs complets en cas d'erreur (automatiquement créés)

## 🔧 Commandes de débogage

### Tester un seul horizon avec logs détaillés

```bash
# Tester J-7 pour 6N8
cd predictTo
python predictTo_train_model.py --hotel 6N8 --horizon 7 --no-azure 2>&1 | tee debug_6N8_J7.log
```

### Vérifier que les données existent

```bash
# Vérifier les fichiers nécessaires
ls -la ../cluster/results/6N8/
ls -la ../data/6N8/

# Vérifier le contenu des CSV
head ../cluster/results/6N8/clustering_results.csv
head ../data/6N8/Indicateurs.csv
```

### Vérifier les colonnes dans les CSV

```python
import pandas as pd

# Vérifier les colonnes de clustering
df_cluster = pd.read_csv('../cluster/results/6N8/clustering_results.csv', sep=';')
print("Colonnes clustering:", df_cluster.columns.tolist())
print("Colonnes TO:", [c for c in df_cluster.columns if c.startswith('J-')])

# Vérifier les colonnes d'indicateurs
df_ind = pd.read_csv('../data/6N8/Indicateurs.csv', sep=';')
print("Colonnes indicateurs:", df_ind.columns.tolist())
print("HotCodes uniques:", df_ind['hotCode'].unique())
```

## 💡 Conseils de débogage

1. **Toujours commencer par un seul horizon** pour identifier rapidement le problème
   ```bash
   python predictTo_train_model.py --hotel 6N8 --horizon 7 --no-azure
   ```

2. **Vérifier les prérequis** avant le batch :
   - Clustering exécuté (`cluster/results/{hotel}/`)
   - Données disponibles (`data/{hotel}/`)
   - Fichiers au bon format (séparateur `;`)

3. **Consulter les logs** dans l'ordre :
   - Terminal (pour l'erreur immédiate)
   - `error_logs/` (pour les détails complets)
   - `predictTo_training.log` (pour le contexte)

4. **Tester sur un hôtel qui fonctionne** (ex: D09) avant de déboguer un hôtel problématique

## 📞 Support

Si l'erreur persiste après avoir suivi ce guide :

1. **Collecter les informations** :
   - Message d'erreur complet
   - Fichier `error_logs/error_{hotel}_J-{horizon}_{timestamp}.log`
   - Commande exécutée
   - Résultats de vérification des fichiers

2. **Vérifier la structure des données** :
   ```bash
   # Extraire les 10 premières lignes
   head -n 10 ../cluster/results/6N8/clustering_results.csv > sample_cluster.csv
   head -n 10 ../data/6N8/Indicateurs.csv > sample_indicateurs.csv
   ```

3. **Créer un rapport d'erreur** avec ces éléments

