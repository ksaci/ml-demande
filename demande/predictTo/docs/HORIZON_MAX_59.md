# Pourquoi l'horizon maximum est J-59 ?

## 🎯 Question

Pourquoi ne peut-on pas entraîner un modèle pour **J-60** alors que les données vont jusqu'à **J-60** ?

## 💡 Réponse

### Principe de base

Pour prédire à un horizon **J-H**, le modèle doit utiliser des données disponibles **J-(H+1) jours avant** (ou plus loin dans le passé).

```
┌────────────────────────────────────────────────────────┐
│  Pour prédire à J-H, on utilise les données de:       │
│  J-(H+1), J-(H+2), ..., J-60                          │
└────────────────────────────────────────────────────────┘
```

### Exemple concret : J-7

Pour prédire le TO du **15 janvier** (date du séjour) avec un horizon de **J-7** :

- **Date d'observation** : 8 janvier (J-7 avant le séjour)
- **Données disponibles** : Jusqu'au 8 janvier
- **Features utilisées** :
  - `pm_J-8` : PM observé le 8 janvier (pour séjour du 15)
  - `pm_J-9` : PM observé le 7 janvier (pour séjour du 15)
  - ...
  - `pm_J-60` : PM observé 60 jours avant le séjour

✅ **Ça marche** car on a 53 colonnes de features (J-8 à J-60)

### Problème avec J-60

Pour prédire le TO du **15 janvier** avec un horizon de **J-60** :

- **Date d'observation** : 16 novembre (J-60 avant le séjour)
- **Données disponibles** : Jusqu'au 16 novembre
- **Features nécessaires** :
  - `pm_J-61` : PM observé 61 jours avant → ❌ **N'existe pas !**
  - `pm_J-62` : PM observé 62 jours avant → ❌ **N'existe pas !**
  - ...

❌ **Ça ne marche PAS** car nos données s'arrêtent à J-60

### Solution : J-59 maximum

Pour prédire le TO avec un horizon de **J-59** :

- **Date d'observation** : 59 jours avant le séjour
- **Features utilisées** :
  - `pm_J-60` : PM observé 60 jours avant ✅ **Existe !**

✅ **Ça marche** avec 1 seule colonne de features historiques (J-60)

## 📊 Tableau récapitulatif

| Horizon | Features PM requises | Disponibles ? | Nombre de features |
|---------|---------------------|---------------|-------------------|
| J-0 | pm_J-1 à pm_J-60 | ✅ Oui | 60 |
| J-7 | pm_J-8 à pm_J-60 | ✅ Oui | 53 |
| J-30 | pm_J-31 à pm_J-60 | ✅ Oui | 30 |
| J-45 | pm_J-46 à pm_J-60 | ✅ Oui | 15 |
| J-59 | pm_J-60 | ✅ Oui | 1 |
| J-60 | pm_J-61 à pm_J-120 | ❌ **Non** | 0 → ❌ **Plantage** |

## 🔧 Solution implémentée

1. **Validation dans le code** : Le script refuse horizon >= 60
2. **Batch par défaut** : Horizons = `[59, 45, 30, 21, 15, 10, 7, 5, 3, 1, 0]`
3. **Message d'erreur clair** :

```bash
python predictTo_train_model.py --hotel D09 --horizon 60
# ❌ L'horizon maximum est 59 (car les données vont jusqu'à J-60)
#    Pour prédire à J-60, il faudrait des données jusqu'à J-61 minimum
```

## 💭 Et si on voulait vraiment J-60 ?

### Option 1 : Étendre les données historiques

Récupérer/générer des données PM/TO jusqu'à J-120 (ou plus) :
- ✅ Permettrait J-60, J-90, etc.
- ❌ Nécessite plus de stockage et calculs
- ❌ Données très anciennes moins pertinentes

### Option 2 : Modèle sans features historiques

Pour J-60, utiliser **uniquement** les features calendaires :
- Mois, jour de la semaine
- Jours fériés, vacances scolaires
- TO de l'année précédente (ToF1)
- ✅ Possible techniquement
- ❌ Performances probablement très faibles (R² < 0.5)

### Option 3 : Accepter J-59 comme maximum

C'est le choix fait actuellement :
- ✅ Simple et cohérent
- ✅ J-59 ≈ 2 mois à l'avance (suffisant pour la plupart des usages)
- ✅ Données de meilleure qualité (pas trop anciennes)

## 🎯 En résumé

```
Données disponibles : J-0 à J-60 (61 points)
                              ↓
Maximum utilisable comme feature : J-60
                              ↓
Horizon maximum supporté : J-59
                              ↓
Pour J-59, on utilise uniquement pm_J-60 comme feature historique
```

**Règle d'or** : Pour prédire à J-H, il faut des données jusqu'à J-(H+1) minimum.

## 📞 Questions fréquentes

**Q : Pourquoi ne pas extrapoler/interpoler les données manquantes ?**  
R : Ce serait créer de fausses données, ce qui biaiserait le modèle.

**Q : J-59 ce n'est pas un peu bizarre comme nombre ?**  
R : C'est une contrainte technique (données jusqu'à J-60). En pratique, J-59 ≈ 2 mois, ce qui est déjà très lointain pour des prédictions hôtelières.

**Q : Peut-on changer la fenêtre de données (ex: J-0 à J-90) ?**  
R : Oui, mais cela nécessite de modifier les données sources (fichiers Indicateurs.csv et rateShopper.csv) pour inclure plus de jours historiques.

