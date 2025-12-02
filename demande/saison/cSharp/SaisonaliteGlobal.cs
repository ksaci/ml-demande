using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace SaisonaliteAnalysis
{
    /// <summary>
    /// Classe pour représenter une période de saison
    /// </summary>
    public class SeasonPeriod
    {
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string SeasonName { get; set; }
        
        public override string ToString()
        {
            return $"{StartDate:yyyy-MM-dd} - {EndDate:yyyy-MM-dd} : {SeasonName}";
        }
    }

    /// <summary>
    /// Classe pour les données d'entrée du clustering
    /// </summary>
    public class SeasonData
    {
        [VectorType(1)]
        public float[] Value { get; set; }
    }

    /// <summary>
    /// Classe pour les prédictions du clustering
    /// </summary>
    public class SeasonPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint ClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }

    /// <summary>
    /// Analyseur de saisonnalité basé sur des données historiques
    /// </summary>
    public class SaisonaliteGlobalAnalyzer
    {
        private readonly string _hotelCode;
        private readonly string _dataPath;
        private readonly int _baseYear;
        private readonly MLContext _mlContext;

        public SaisonaliteGlobalAnalyzer(string hotelCode, string dataPath = "../../data/", int baseYear = 2025)
        {
            _hotelCode = hotelCode;
            _dataPath = dataPath;
            _baseYear = baseYear;
            _mlContext = new MLContext(seed: 42);
        }

        /// <summary>
        /// Point d'entrée principal de l'analyse
        /// </summary>
        public List<SeasonPeriod> AnalyzeSeasonality(string metric = "To")
        {
            Console.WriteLine("\n===== ANALYSE DE SAISONNALITÉ =====\n");
            
            // 1. Chargement des données
            var data = LoadData();
            Console.WriteLine($"Données chargées : {data.Count} lignes");

            // 2. Construire le profil par jour de l'année
            var profile = BuildDayOfYearProfile(data, metric);
            Console.WriteLine($"Profil construit : {profile.Length} jours");

            // 3. Lissage Savitzky-Golay
            var smoothed = SavitzkyGolayFilter(profile, 21, 3);
            Console.WriteLine("Lissage appliqué");

            // 4. Normalisation 0-100
            var normalized = MinMaxNormalize(smoothed, 0, 100);
            Console.WriteLine("Normalisation effectuée");

            // 5. Clustering KMeans
            var labels = PerformKMeansClustering(normalized, 3);
            Console.WriteLine("Clustering effectué");

            // 6. Extraction des périodes
            var periods = ExtractPeriods(labels);
            Console.WriteLine($"Périodes extraites : {periods.Count}");

            // 7. Fusion des segments courts
            var cleanPeriods = MergeShortSegments(periods, 14);
            Console.WriteLine($"Périodes nettoyées : {cleanPeriods.Count}");

            // 8. Conversion en dates réelles
            var seasonPeriods = ConvertToDatePeriods(cleanPeriods);

            // 9. Affichage des résultats
            Console.WriteLine("\n===== SAISONS CALCULÉES =====\n");
            foreach (var period in seasonPeriods)
            {
                Console.WriteLine(period);
            }

            return seasonPeriods;
        }

        /// <summary>
        /// Charge les données depuis le fichier CSV
        /// </summary>
        private List<Dictionary<string, string>> LoadData()
        {
            var filePath = Path.Combine(_dataPath, _hotelCode, "Indicateurs.csv");
            var data = new List<Dictionary<string, string>>();

            using (var reader = new StreamReader(filePath))
            {
                var headers = reader.ReadLine()?.Split(';');
                if (headers == null) return data;

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line?.Split(';');
                    if (values == null) continue;

                    var row = new Dictionary<string, string>();
                    for (int i = 0; i < headers.Length && i < values.Length; i++)
                    {
                        row[headers[i]] = string.IsNullOrEmpty(values[i]) ? "0" : values[i];
                    }
                    data.Add(row);
                }
            }

            return data;
        }

        /// <summary>
        /// Construit un profil moyen par jour de l'année (1-365)
        /// </summary>
        private double[] BuildDayOfYearProfile(List<Dictionary<string, string>> data, string metric)
        {
            var dayValues = new Dictionary<int, List<double>>();

            foreach (var row in data)
            {
                if (DateTime.TryParse(row["Date"], out var date))
                {
                    int dayOfYear = date.DayOfYear;
                    
                    if (double.TryParse(row[metric], NumberStyles.Any, CultureInfo.InvariantCulture, out var value))
                    {
                        if (!dayValues.ContainsKey(dayOfYear))
                            dayValues[dayOfYear] = new List<double>();
                        
                        dayValues[dayOfYear].Add(value);
                    }
                }
            }

            // Calculer la moyenne pour chaque jour et interpoler les manquants
            var profile = new double[365];
            for (int day = 1; day <= 365; day++)
            {
                if (dayValues.ContainsKey(day) && dayValues[day].Count > 0)
                {
                    profile[day - 1] = dayValues[day].Average();
                }
                else
                {
                    // Interpolation linéaire simple
                    profile[day - 1] = InterpolateValue(profile, day - 1, dayValues);
                }
            }

            return profile;
        }

        /// <summary>
        /// Interpolation simple pour les valeurs manquantes
        /// </summary>
        private double InterpolateValue(double[] profile, int index, Dictionary<int, List<double>> dayValues)
        {
            // Chercher la valeur précédente et suivante
            int prev = -1, next = -1;
            
            for (int i = index; i >= 0; i--)
            {
                if (dayValues.ContainsKey(i + 1) && dayValues[i + 1].Count > 0)
                {
                    prev = i;
                    break;
                }
            }

            for (int i = index; i < 365; i++)
            {
                if (dayValues.ContainsKey(i + 1) && dayValues[i + 1].Count > 0)
                {
                    next = i;
                    break;
                }
            }

            if (prev >= 0 && next >= 0)
            {
                double ratio = (double)(index - prev) / (next - prev);
                return profile[prev] + ratio * (profile[next] - profile[prev]);
            }
            else if (prev >= 0)
            {
                return profile[prev];
            }
            else if (next >= 0)
            {
                return profile[next];
            }

            return 0;
        }

        /// <summary>
        /// Filtre de Savitzky-Golay pour le lissage
        /// Implémentation simplifiée
        /// </summary>
        private double[] SavitzkyGolayFilter(double[] data, int windowLength, int polynomialOrder)
        {
            // Pour une implémentation complète, utilisez Math.NET Numerics
            // Ici, on utilise une moyenne mobile pondérée comme approximation
            var result = new double[data.Length];
            int halfWindow = windowLength / 2;

            for (int i = 0; i < data.Length; i++)
            {
                double sum = 0;
                int count = 0;

                for (int j = -halfWindow; j <= halfWindow; j++)
                {
                    int index = i + j;
                    if (index >= 0 && index < data.Length)
                    {
                        // Poids gaussien simple
                        double weight = Math.Exp(-(j * j) / (2.0 * (windowLength / 4.0)));
                        sum += data[index] * weight;
                        count++;
                    }
                }

                result[i] = sum / count;
            }

            return result;
        }

        /// <summary>
        /// Normalisation Min-Max
        /// </summary>
        private double[] MinMaxNormalize(double[] data, double min, double max)
        {
            double dataMin = data.Min();
            double dataMax = data.Max();
            double range = dataMax - dataMin;

            return data.Select(x => min + ((x - dataMin) / range) * (max - min)).ToArray();
        }

        /// <summary>
        /// Clustering KMeans avec ML.NET
        /// </summary>
        private int[] PerformKMeansClustering(double[] data, int numClusters)
        {
            // Préparer les données pour ML.NET
            var mlData = data.Select(x => new SeasonData 
            { 
                Value = new float[] { (float)x } 
            }).ToList();

            var dataView = _mlContext.Data.LoadFromEnumerable(mlData);

            // Définir le pipeline de clustering
            var pipeline = _mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Value",
                numberOfClusters: numClusters);

            // Entraîner le modèle
            var model = pipeline.Fit(dataView);

            // Prédire les clusters
            var predictions = model.Transform(dataView);
            var predictedLabels = _mlContext.Data.CreateEnumerable<SeasonPrediction>(predictions, false).ToArray();

            // Extraire les centres des clusters
            var centroids = new Dictionary<uint, double>();
            for (int i = 0; i < data.Length; i++)
            {
                var clusterId = predictedLabels[i].ClusterId;
                if (!centroids.ContainsKey(clusterId))
                {
                    centroids[clusterId] = data[i];
                }
            }

            // Ordonner les clusters du plus faible au plus fort
            var orderedClusters = centroids.OrderBy(x => x.Value).Select(x => x.Key).ToArray();
            var labelMap = new Dictionary<uint, int>();
            for (int i = 0; i < orderedClusters.Length; i++)
            {
                labelMap[orderedClusters[i]] = i;
            }

            // Mapper les labels
            return predictedLabels.Select(p => labelMap[p.ClusterId]).ToArray();
        }

        /// <summary>
        /// Extraction des périodes continues
        /// </summary>
        private List<(int start, int end, int label)> ExtractPeriods(int[] labels)
        {
            var periods = new List<(int, int, int)>();
            int currentSeason = labels[0];
            int startDay = 1;

            for (int day = 2; day <= 365; day++)
            {
                if (labels[day - 1] != currentSeason)
                {
                    periods.Add((startDay, day - 1, currentSeason));
                    startDay = day;
                    currentSeason = labels[day - 1];
                }
            }

            periods.Add((startDay, 365, currentSeason));
            return periods;
        }

        /// <summary>
        /// Fusion des segments trop courts
        /// </summary>
        private List<(int start, int end, int label)> MergeShortSegments(
            List<(int start, int end, int label)> periods, int minLength)
        {
            var merged = new List<(int start, int end, int label)>();

            for (int i = 0; i < periods.Count; i++)
            {
                var (start, end, label) = periods[i];
                int length = end - start + 1;

                if (length < minLength)
                {
                    // Fusionner avec le segment précédent si possible
                    if (merged.Count > 0)
                    {
                        var prev = merged[merged.Count - 1];
                        merged[merged.Count - 1] = (prev.start, end, prev.label);
                    }
                    else if (i + 1 < periods.Count)
                    {
                        // Fusionner avec le suivant
                        var next = periods[i + 1];
                        merged.Add((start, next.end, next.label));
                        i++; // Sauter le suivant
                    }
                }
                else
                {
                    merged.Add((start, end, label));
                }
            }

            return merged;
        }

        /// <summary>
        /// Conversion des jours de l'année en dates réelles
        /// </summary>
        private List<SeasonPeriod> ConvertToDatePeriods(List<(int start, int end, int label)> periods)
        {
            var seasonNames = new[] { "Basse", "Moyenne", "Haute" };
            var result = new List<SeasonPeriod>();

            foreach (var (start, end, label) in periods)
            {
                var startDate = new DateTime(_baseYear, 1, 1).AddDays(start - 1);
                var endDate = new DateTime(_baseYear, 1, 1).AddDays(end - 1);
                
                result.Add(new SeasonPeriod
                {
                    StartDate = startDate,
                    EndDate = endDate,
                    SeasonName = seasonNames[label]
                });
            }

            return result;
        }

        /// <summary>
        /// Sauvegarde les résultats dans un fichier CSV
        /// </summary>
        public void SaveResults(List<SeasonPeriod> periods, string outputPath)
        {
            using (var writer = new StreamWriter(outputPath))
            {
                writer.WriteLine("Début;Fin;Saison");
                foreach (var period in periods)
                {
                    writer.WriteLine($"{period.StartDate:yyyy-MM-dd};{period.EndDate:yyyy-MM-dd};{period.SeasonName}");
                }
            }
            
            Console.WriteLine($"\nRésultats sauvegardés dans : {outputPath}");
        }
    }

    /// <summary>
    /// Programme principal
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // Configuration
                string hotelCode = "BEE";
                string metric = "To";
                int baseYear = 2025;

                // Créer l'analyseur
                var analyzer = new SaisonaliteGlobalAnalyzer(hotelCode, "../../data/", baseYear);

                // Analyser la saisonnalité
                var seasons = analyzer.AnalyzeSeasonality(metric);

                // Sauvegarder les résultats
                string outputPath = $"../courbe_saison_{hotelCode.ToLower()}_365.csv";
                analyzer.SaveResults(seasons, outputPath);

                Console.WriteLine("\n===== ANALYSE TERMINÉE =====");
                Console.WriteLine($"\nNombre de périodes : {seasons.Count}");
                
                // Statistiques par saison
                var stats = seasons.GroupBy(s => s.SeasonName)
                    .Select(g => new 
                    { 
                        Saison = g.Key, 
                        NbPeriodes = g.Count(),
                        TotalJours = g.Sum(s => (s.EndDate - s.StartDate).Days + 1)
                    });

                Console.WriteLine("\nStatistiques :");
                foreach (var stat in stats)
                {
                    Console.WriteLine($"  {stat.Saison}: {stat.NbPeriodes} période(s), {stat.TotalJours} jour(s)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERREUR : {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
    }
}

