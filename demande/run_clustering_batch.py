"""
Script pour analyser plusieurs hôtels en batch.

Usage:
    python run_clustering_batch.py
    
Le script analyse tous les hôtels listés dans la variable HOTELS.
Modifiez cette liste selon vos besoins.
"""

import os
import sys
from datetime import datetime
from prediction_cluster import HotelBookingClustering

# Liste des codes hôtel à analyser
HOTELS = ['D09', 'A12', 'B05', 'C23']

# Configuration commune
DAYS_BEFORE = 60
YEAR_FILTER = None
ENABLE_SMOOTHING = True
SMOOTHING_WINDOW = 15
SMOOTHING_POLYORDER = 3

# Options de clustering
N_CLUSTERS = 10  # Nombre de clusters (par défaut : 10)
AUTO_FIND_K = False  # Recherche automatique du nombre optimal (True pour activer)
USE_DTW = False  # True = DTW (meilleure qualité, lent) | False = Euclidean (rapide, dev)


def analyze_hotel(hotCode):
    """
    Analyse un hôtel spécifique.
    
    Args:
        hotCode (str): Code de l'hôtel à analyser
    
    Returns:
        bool: True si succès, False si échec
    """
    try:
        print(f"\n{'='*80}")
        print(f"🏨 Traitement de l'hôtel : {hotCode}")
        print(f"{'='*80}\n")
        
        # Vérifier que le fichier existe
        data_file = f'data/{hotCode}/Indicateurs.csv'
        if not os.path.exists(data_file):
            print(f"⚠️  ATTENTION : Fichier non trouvé : {data_file}")
            print(f"    → Hôtel {hotCode} ignoré\n")
            return False
        
        start_time = datetime.now()
        
        # Créer l'instance
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
            print(f"\n💡 Recherche automatique du nombre optimal de clusters...")
            optimal_k = clustering.find_optimal_clusters(max_k=10, metric="euclidean")
            print(f"✓ K optimal : {optimal_k}")
        else:
            optimal_k = N_CLUSTERS
            print(f"\n💡 Utilisation de {optimal_k} clusters (configuré)")
        
        # Effectuer le clustering
        if USE_DTW:
            metric = "dtw"
            n_init = 5
        else:
            metric = "euclidean"
            n_init = 10
        
        clustering.perform_clustering(n_clusters=optimal_k, metric=metric, n_init=n_init)
        
        # Visualiser
        clustering.visualize_clusters()
        
        # Analyser les caractéristiques
        clustering.analyze_cluster_characteristics()
        
        # Sauvegarder les résultats
        clustering.save_results()
        clustering.save_model()
        clustering.save_cluster_profiles()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"✓ Hôtel {hotCode} terminé avec succès !")
        print(f"  Durée : {duration}")
        print(f"  Résultats : results/{hotCode}/")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ ERREUR lors du traitement de l'hôtel {hotCode}")
        print(f"{'='*80}")
        print(f"Type d'erreur : {type(e).__name__}")
        print(f"Message : {str(e)}")
        print(f"{'='*80}\n")
        return False


def main():
    """
    Fonction principale : analyse tous les hôtels en batch.
    """
    print("="*80)
    print("  ANALYSE DE CLUSTERING EN BATCH - PLUSIEURS HÔTELS")
    print("="*80)
    print()
    
    total_hotels = len(HOTELS)
    success_count = 0
    failed_count = 0
    failed_hotels = []
    
    print(f"📋 Hôtels à traiter : {total_hotels}")
    print(f"    {', '.join(HOTELS)}")
    print()
    
    overall_start = datetime.now()
    
    # Traiter chaque hôtel
    for index, hotel in enumerate(HOTELS, start=1):
        print(f"\n{'='*80}")
        print(f"  [{index}/{total_hotels}] Hôtel : {hotel}")
        print(f"{'='*80}")
        
        success = analyze_hotel(hotel)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_hotels.append(hotel)
    
    overall_end = datetime.now()
    overall_duration = overall_end - overall_start
    
    # Résumé final
    print("\n" + "="*80)
    print("  RÉSUMÉ DE L'ANALYSE EN BATCH")
    print("="*80)
    print()
    print(f"Total d'hôtels traités : {total_hotels}")
    print(f"✓ Succès : {success_count}")
    print(f"❌ Échecs : {failed_count}")
    
    if failed_count > 0:
        print()
        print(f"Hôtels en échec : {', '.join(failed_hotels)}")
    
    print()
    print(f"⏱️  Durée totale : {overall_duration}")
    print()
    print("📁 Les résultats sont disponibles dans : results/")
    print()
    
    # Afficher les dossiers de résultats créés
    print("Dossiers de résultats créés :")
    for hotel in HOTELS:
        result_dir = f'results/{hotel}'
        if os.path.exists(result_dir):
            file_count = len([f for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f))])
            print(f"  ✓ {result_dir} ({file_count} fichiers)")
    
    print()
    print("="*80)
    print()
    
    # Code de sortie
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

