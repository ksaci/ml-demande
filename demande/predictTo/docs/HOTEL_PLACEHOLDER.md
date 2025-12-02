# Utilisation du placeholder {hotCode} dans la configuration

## 🎯 Problème résolu

**Avant** : Les chemins dans `config_predictTo.yaml` étaient codés en dur avec `D09` :
```yaml
data:
  indicateurs: "../data/D09/Indicateurs.csv"
  rateShopper: "../data/D09/rateShopper.csv"
```

❌ **Problème** : Pour entraîner un modèle pour l'hôtel `6N8`, il fallait modifier manuellement le fichier de config !

## ✅ Solution implémentée

Les chemins utilisent maintenant le placeholder `{hotCode}` qui est automatiquement remplacé par le code de l'hôtel :

```yaml
data:
  clustering_results: "../cluster/results/{hotCode}/clustering_results.csv"
  indicateurs: "../data/{hotCode}/Indicateurs.csv"
  rateShopper: "../data/{hotCode}/rateShopper.csv"
```

## 🔧 Comment ça fonctionne

### 1. **Configuration avec placeholder**

Dans `config_predictTo.yaml` :
```yaml
data:
  indicateurs: "../data/{hotCode}/Indicateurs.csv"
```

### 2. **Remplacement automatique**

Quand vous lancez avec `--hotel 6N8` :
```bash
python predictTo_train_model.py --hotel 6N8 --horizon 7
```

Le code remplace automatiquement `{hotCode}` par `6N8` :
```
../data/{hotCode}/Indicateurs.csv  →  ../data/6N8/Indicateurs.csv
```

### 3. **Résultat**

Les fichiers suivants sont chargés :
- ✅ `../cluster/results/6N8/clustering_results.csv`
- ✅ `../data/6N8/Indicateurs.csv`
- ✅ `../data/6N8/rateShopper.csv`

Et les résultats sont sauvegardés dans :
- ✅ `results/6N8/J-7/models/`
- ✅ Azure: `ml-models/predictTo/6N8/J-7/`

## 📊 Exemples concrets

### Exemple 1 : Hôtel D09

```bash
python predictTo_train_model.py --hotel D09 --horizon 7
```

**Chemins résolus** :
- Clustering : `../cluster/results/D09/clustering_results.csv`
- Indicateurs : `../data/D09/Indicateurs.csv`
- RateShopper : `../data/D09/rateShopper.csv`
- Sortie : `results/D09/J-7/`

### Exemple 2 : Hôtel 6N8

```bash
python predictTo_train_model.py --hotel 6N8 --horizon 14
```

**Chemins résolus** :
- Clustering : `../cluster/results/6N8/clustering_results.csv`
- Indicateurs : `../data/6N8/Indicateurs.csv`
- RateShopper : `../data/6N8/rateShopper.csv`
- Sortie : `results/6N8/J-14/`

### Exemple 3 : Hôtel 0BT

```bash
python predictTo_train_model.py --hotel 0BT --horizon 30
```

**Chemins résolus** :
- Clustering : `../cluster/results/0BT/clustering_results.csv`
- Indicateurs : `../data/0BT/Indicateurs.csv`
- RateShopper : `../data/0BT/rateShopper.csv`
- Sortie : `results/0BT/J-30/`

## 🔍 Détails techniques

### Code de remplacement

Dans la classe `XGBoostOccupancyPredictor` :

```python
def _replace_hotel_placeholder(self):
    """Remplace {hotCode} par le code d'hôtel réel."""
    if not self.hotel_code:
        return
    
    paths_to_replace = [
        'clustering_results_path',
        'indicateurs_path',
        'rateShopper_path'
    ]
    
    for path_key in paths_to_replace:
        if path_key in self.config:
            original_path = self.config[path_key]
            if '{hotCode}' in original_path:
                new_path = original_path.replace('{hotCode}', self.hotel_code)
                self.config[path_key] = new_path
```

### Moment du remplacement

Le remplacement se fait dans `__init__()`, **avant** le chargement des données :

1. Configuration chargée depuis YAML
2. `_replace_hotel_placeholder()` appelé ✅
3. `load_data()` utilise les chemins corrigés

## 🎯 Avantages

### 1. **Un seul fichier de configuration**
Plus besoin de créer `config_D09.yaml`, `config_6N8.yaml`, etc.

### 2. **Batch training simplifié**
```bash
# Entraîner plusieurs hôtels facilement
python run_predictTo_batch.py --hotel D09
python run_predictTo_batch.py --hotel 6N8
python run_predictTo_batch.py --hotel 0BT
```

### 3. **Moins d'erreurs**
Impossible d'oublier de changer un chemin dans la config !

### 4. **Structure cohérente**
```
data/
├── D09/
│   ├── Indicateurs.csv
│   └── rateShopper.csv
├── 6N8/
│   ├── Indicateurs.csv
│   └── rateShopper.csv
└── 0BT/
    ├── Indicateurs.csv
    └── rateShopper.csv
```

Tous les hôtels suivent la même structure.

## 📝 Mode global (sans --hotel)

Si vous n'utilisez **PAS** `--hotel`, le placeholder `{hotCode}` **n'est pas remplacé** :

```bash
# Sans --hotel (mode global, peu recommandé)
python predictTo_train_model.py --horizon 7
```

Dans ce cas, le code utilise directement les chemins de la config :
```yaml
clustering_results: "../cluster/results/{hotCode}/clustering_results.csv"
```

Le fichier cherché sera littéralement `../cluster/results/{hotCode}/clustering_results.csv` → ❌ **Plantage !**

**Recommandation** : Toujours utiliser `--hotel` pour un entraînement spécifique à un hôtel.

## 🧪 Tester le remplacement

Exécutez le script de test :

```bash
cd predictTo
python test_hotel_config.py
```

Résultat :
```
✅ HÔTEL: 6N8
   clustering_results_path:
   Avant: ../cluster/results/{hotCode}/clustering_results.csv
   Après: ../cluster/results/6N8/clustering_results.csv
   
   indicateurs_path:
   Avant: ../data/{hotCode}/Indicateurs.csv
   Après: ../data/6N8/Indicateurs.csv
```

## 🔧 Personnalisation

Si vous avez une structure de dossiers différente, modifiez simplement `config_predictTo.yaml` :

```yaml
data:
  # Structure custom
  indicateurs: "/mon/chemin/custom/{hotCode}/data/Indicateurs.csv"
  rateShopper: "/autre/chemin/{hotCode}/rateShopper.csv"
```

Le placeholder `{hotCode}` sera toujours remplacé automatiquement !

## 📞 Questions fréquentes

**Q : Puis-je utiliser `{hotCode}` dans d'autres chemins ?**  
R : Actuellement, seuls `clustering_results_path`, `indicateurs_path` et `rateShopper_path` sont supportés. Pour ajouter d'autres chemins, modifiez la liste dans `_replace_hotel_placeholder()`.

**Q : Que se passe-t-il si le dossier de l'hôtel n'existe pas ?**  
R : Le script affichera une erreur claire `FileNotFoundError` avec le chemin complet manquant.

**Q : Peut-on avoir plusieurs placeholders (ex: `{hotCode}` et `{year}`) ?**  
R : Pas actuellement, mais le code peut être étendu facilement pour supporter d'autres placeholders.

## 🎉 En résumé

✅ Un seul fichier de config pour tous les hôtels  
✅ Remplacement automatique de `{hotCode}`  
✅ Structure cohérente et maintenable  
✅ Moins d'erreurs humaines  
✅ Batch training simplifié  

Le placeholder `{hotCode}` rend la configuration **dynamique** et **réutilisable** ! 🚀

