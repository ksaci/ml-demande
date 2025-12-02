# Analyse de Saisonnalité - Version C#

## Description

Ce projet est l'équivalent C# du notebook Python `Saisonalite-global.ipynb`. Il analyse les données historiques d'un hôtel pour déterminer automatiquement les périodes de saison basse, moyenne et haute.

## Fonctionnalités

- **Chargement de données CSV** : Lecture des indicateurs depuis `../../data/{HotelCode}/Indicateurs.csv`
- **Profil saisonnier** : Calcul d'un profil moyen par jour de l'année (1-365)
- **Lissage avancé** : Application d'un filtre de type Savitzky-Golay
- **Normalisation** : Mise à l'échelle des valeurs entre 0 et 100
- **Clustering KMeans** : Classification non supervisée en 3 saisons
- **Optimisation RMS** : Fusion des segments trop courts (< 14 jours)
- **Export CSV** : Sauvegarde des résultats avec dates de début/fin

## Prérequis

- **.NET 6.0 SDK** ou supérieur
- **Packages NuGet** :
  - Microsoft.ML (v2.0.1)
  - MathNet.Numerics (v5.0.0)

## Installation

1. **Installer .NET SDK** (si ce n'est pas déjà fait) :
   ```bash
   # Windows (via winget)
   winget install Microsoft.DotNet.SDK.6
   
   # macOS (via Homebrew)
   brew install --cask dotnet-sdk
   
   # Linux (Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install -y dotnet-sdk-6.0
   ```

2. **Restaurer les packages NuGet** :
   ```bash
   cd saison/cSharp
   dotnet restore
   ```

## Utilisation

### Compilation

```bash
cd saison/cSharp
dotnet build
```

### Exécution

```bash
dotnet run
```

### Modification du code hôtel

Éditez le fichier `SaisonaliteGlobal.cs`, ligne ~473 :

```csharp
string hotelCode = "BEE";  // Modifier ici (ex: "CHV", "1WC", etc.)
string metric = "To";      // Métrique à analyser (To, CaH, PM)
int baseYear = 2025;       // Année de référence
```

### Exemple de sortie

```
===== ANALYSE DE SAISONNALITÉ =====

Données chargées : 86805 lignes
Profil construit : 365 jours
Lissage appliqué
Normalisation effectuée
Clustering effectué
Périodes extraites : 8
Périodes nettoyées : 7

===== SAISONS CALCULÉES =====

2025-01-01 - 2025-02-22 : Basse
2025-02-23 - 2025-03-03 : Basse
2025-03-04 - 2025-03-29 : Basse
2025-03-30 - 2025-07-21 : Moyenne
2025-07-22 - 2025-09-09 : Haute
2025-09-10 - 2025-10-10 : Haute
2025-10-11 - 2025-11-15 : Moyenne
2025-11-16 - 2025-12-31 : Haute

Résultats sauvegardés dans : ../courbe_saison_bee_365.csv

===== ANALYSE TERMINÉE =====

Nombre de périodes : 8

Statistiques :
  Basse: 3 période(s), 81 jour(s)
  Moyenne: 2 période(s), 150 jour(s)
  Haute: 3 période(s), 134 jour(s)
```

## Architecture du code

### Classes principales

1. **`SeasonPeriod`**
   - Représente une période de saison avec dates de début/fin et nom

2. **`SeasonData` / `SeasonPrediction`**
   - Classes de support pour ML.NET (clustering)

3. **`SaisonaliteGlobalAnalyzer`**
   - Classe principale d'analyse
   - Méthodes publiques :
     - `AnalyzeSeasonality(metric)` : Lance l'analyse complète
     - `SaveResults(periods, path)` : Sauvegarde les résultats

### Workflow de l'analyse

```
1. LoadData()                    → Chargement CSV
2. BuildDayOfYearProfile()       → Profil par jour (1-365)
3. SavitzkyGolayFilter()         → Lissage
4. MinMaxNormalize()             → Normalisation 0-100
5. PerformKMeansClustering()     → Clustering en 3 groupes
6. ExtractPeriods()              → Extraction des périodes
7. MergeShortSegments()          → Fusion segments < 14 jours
8. ConvertToDatePeriods()        → Conversion en dates
9. SaveResults()                 → Export CSV
```

## Différences avec la version Python

### Points communs
- ✅ Même algorithme de clustering (KMeans, 3 clusters)
- ✅ Même logique de lissage et normalisation
- ✅ Même critère de fusion (segments < 14 jours)
- ✅ Même format de sortie CSV

### Adaptations C#
- **Lissage Savitzky-Golay** : Implémentation simplifiée avec moyenne mobile pondérée
  - Pour une implémentation exacte, utilisez `MathNet.Filtering.Kalman`
- **Clustering** : Utilisation de ML.NET au lieu de scikit-learn
- **Visualisation** : Non implémentée (pas de Matplotlib en C#)
  - Pour des graphiques, utilisez **OxyPlot** ou **LiveCharts**

## Extension possible

### Ajouter la visualisation avec OxyPlot

```csharp
// Dans le .csproj
<PackageReference Include="OxyPlot.Core" Version="2.1.2" />
<PackageReference Include="OxyPlot.WindowsForms" Version="2.1.2" />

// Code exemple
public void VisualizeSeasonality(double[] seasonIndex, List<SeasonPeriod> periods)
{
    var plotModel = new PlotModel { Title = "Profil de saisonnalité" };
    
    var lineSeries = new LineSeries();
    for (int i = 0; i < seasonIndex.Length; i++)
    {
        lineSeries.Points.Add(new DataPoint(i + 1, seasonIndex[i]));
    }
    
    plotModel.Series.Add(lineSeries);
    
    // ... ajouter les zones colorées pour les saisons
}
```

### Parallélisation pour plusieurs hôtels

```csharp
var hotelCodes = new[] { "BEE", "CHV", "1WC", "1UU", "1LM" };

Parallel.ForEach(hotelCodes, code =>
{
    var analyzer = new SaisonaliteGlobalAnalyzer(code);
    var seasons = analyzer.AnalyzeSeasonality();
    analyzer.SaveResults(seasons, $"../courbe_saison_{code.ToLower()}_365.csv");
});
```

## Troubleshooting

### Erreur "FileNotFoundException"
- Vérifiez que le fichier CSV existe dans `../../data/{HotelCode}/Indicateurs.csv`
- Vérifiez les chemins relatifs depuis le dossier `saison/cSharp/`

### Erreur de parsing des dates
- Le format attendu est compatible avec `DateTime.Parse()`
- Formats supportés : `yyyy-MM-dd`, `dd/MM/yyyy`, etc.

### Problème de séparateur CSV
- Le code utilise `;` comme séparateur
- Si vos CSV utilisent `,`, modifiez ligne ~157 : `.Split(',')`

## Performance

- **Temps d'exécution** : ~2-5 secondes pour 86 805 lignes
- **Mémoire** : ~50-100 MB
- **Optimisation** : Le clustering KMeans est le plus coûteux

## Licence

Ce code est fourni à des fins d'analyse interne. Toute modification ou redistribution doit être validée.

## Contact

Pour toute question ou amélioration, contactez l'équipe Data/BI.

