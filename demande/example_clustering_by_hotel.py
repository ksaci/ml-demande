"""
Script d'exemple pour exécuter l'analyse de clustering par hôtel.

Ce script montre comment utiliser prediction_cluster.py pour analyser 
les données d'un hôtel spécifique.

Usage:
    python example_clustering_by_hotel.py
    
    Ou en spécifiant le code hôtel directement:
    python example_clustering_by_hotel.py D09
"""

import sys
from prediction_cluster import HotelBookingClustering

def main():
    """
    Exemple d'utilisation de l'analyse de clustering par hôtel.
    """
    
    # 1. Méthode 1 : Spécifier le code hôtel via argument ligne de commande
    if len(sys.argv) > 1:
        hotCode = sys.argv[1].strip().upper()
    else:
        # 2. Méthode 2 : Demander interactivement
        hotCode = input("Entrez le code de l'hôtel (3 caractères, ex: D09) : ").strip().upper()
    
    print("\n" + "="*80)
    print("EXEMPLE : ANALYSE DE CLUSTERING PAR HÔTEL")
    print("="*80)
    print(f"\n🏨 Hôtel : {hotCode}")
    print(f"📂 Données : data/{hotCode}/Indicateurs.csv")
    print(f"💾 Résultats : results/{hotCode}/")
    print()
    
    # Configuration
    DAYS_BEFORE = 60  # J-60 à J
    YEAR_FILTER = None  # Toutes les années (ou spécifier une année, ex: 2024)
    
    # Options de lissage
    ENABLE_SMOOTHING = True
    SMOOTHING_WINDOW = 15
    SMOOTHING_POLYORDER = 3
    
    # Options de clustering
    N_CLUSTERS = 10  # Nombre de clusters (par défaut : 10)
    AUTO_FIND_K = False  # Recherche automatique du nombre optimal (True pour activer)
    USE_DTW = False  # True = DTW (meilleure qualité, lent) | False = Euclidean (rapide, dev)
    
    # Créer l'instance avec le code hôtel
    clustering = HotelBookingClustering(hotCode=hotCode, days_before=DAYS_BEFORE)
    
    # Charger les données
    clustering.load_data(year_filter=YEAR_FILTER)
    
    # Préparer les courbes
    clustering.prepare_booking_curves(min_observations=20)
    
    # Appliquer le lissage
    clustering.apply_smoothing(
        enable=ENABLE_SMOOTHING,
        window_length=SMOOTHING_WINDOW,
        polyorder=SMOOTHING_POLYORDER
    )
    
    # Analyser les taux d'occupation initiaux
    clustering.analyze_initial_occupancy()
    
    # Normaliser
    clustering.normalize_curves()
    
    # Déterminer le nombre de clusters
    if AUTO_FIND_K:
        print(f"\n💡 Recherche automatique du nombre optimal de clusters avec 'euclidean' (rapide)...")
        optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
        print(f"✓ K optimal : {optimal_k}")
    else:
        optimal_k = N_CLUSTERS
        print(f"\n💡 Utilisation de {optimal_k} clusters (configuré)")
        print("   Pour activer la recherche automatique : AUTO_FIND_K = True")
    
    # Effectuer le clustering
    if USE_DTW:
        metric = "dtw"
        n_init = 5
        print(f"\n💡 Clustering final avec DTW et K={optimal_k}...")
        print("   Mode production - meilleure qualité")
    else:
        metric = "euclidean"
        n_init = 10
        print(f"\n💡 Clustering final avec EUCLIDEAN et K={optimal_k}...")
        print("   ⚠️  Mode développement - Changez USE_DTW = True pour la production")
    
    clustering.perform_clustering(n_clusters=optimal_k, metric=metric, n_init=n_init)
    
    # Visualiser
    clustering.visualize_clusters()
    
    # Analyser les caractéristiques
    clustering.analyze_cluster_characteristics()
    
    # Sauvegarder tous les résultats
    clustering.save_results()
    clustering.save_model()
    clustering.save_cluster_profiles()
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS !")
    print("="*80)
    print(f"\n📁 Tous les fichiers ont été sauvegardés dans : results/{hotCode}/")
    print()
    print("💡 Pour prédire le cluster d'une nouvelle courbe :")
    print(f"   clustering = HotelBookingClustering(hotCode='{hotCode}')")
    print("   clustering.load_model()")
    print("   result = clustering.predict_cluster({'J-60': 0.1, 'J-59': 0.12, ...})")
    print()


if __name__ == "__main__":
    main()

