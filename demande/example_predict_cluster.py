"""
Script d'exemple montrant comment utiliser le modèle de clustering 
pour prédire le cluster d'une nouvelle date avec une montée en charge incomplète.

Ce script peut être utilisé dans le modèle d'entraînement de prédiction du To.
"""

import pandas as pd
import numpy as np
from prediction_cluster import HotelBookingClustering

def example_1_predict_with_partial_data():
    """
    Exemple 1 : Prédire le cluster pour une date avec seulement 30 jours de données.
    """
    print("\n" + "="*80)
    print("EXEMPLE 1 : Prédiction avec données partielles (J-60 à J-30)")
    print("="*80)
    
    # Charger le modèle de clustering pré-entrainé
    clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
    clustering.load_model('results/clustering_model.pkl')
    
    # Charger également les données pour avoir les profils
    clustering.load_data()
    clustering.prepare_booking_curves()
    
    # Simuler une courbe incomplète (on a seulement de J-60 à J-30)
    partial_curve = {
        'J-60': 0.15,
        'J-59': 0.16,
        'J-58': 0.17,
        'J-57': 0.18,
        'J-56': 0.19,
        'J-55': 0.20,
        'J-54': 0.21,
        'J-53': 0.22,
        'J-52': 0.23,
        'J-51': 0.24,
        'J-50': 0.25,
        'J-49': 0.26,
        'J-48': 0.27,
        'J-47': 0.28,
        'J-46': 0.29,
        'J-45': 0.30,
        'J-44': 0.31,
        'J-43': 0.32,
        'J-42': 0.33,
        'J-41': 0.34,
        'J-40': 0.35,
        'J-39': 0.36,
        'J-38': 0.37,
        'J-37': 0.38,
        'J-36': 0.39,
        'J-35': 0.40,
        'J-34': 0.41,
        'J-33': 0.42,
        'J-32': 0.43,
        'J-31': 0.44,
        'J-30': 0.45
    }
    
    # Prédire le cluster
    prediction = clustering.predict_cluster(partial_curve)
    
    print(f"\n✓ Cluster prédit : {prediction['cluster']}")
    print(f"  - Confiance : {prediction['confidence']:.3f}")
    
    # Obtenir le profil moyen du cluster prédit
    profile = clustering.get_cluster_profile(prediction['cluster'])
    print(f"\n📊 Profil du cluster {prediction['cluster']} :")
    print(f"  - Nombre d'échantillons : {profile['n_samples']}")
    print(f"  - To moyen à J : {profile['mean_curve']['J-0']:.3f}")
    print(f"  - To médian à J : {profile['median_curve']['J-0']:.3f}")
    
    return prediction, profile


def example_2_load_profiles_from_csv():
    """
    Exemple 2 : Charger les profils moyens depuis le fichier CSV.
    """
    print("\n" + "="*80)
    print("EXEMPLE 2 : Chargement des profils depuis CSV")
    print("="*80)
    
    # Charger les profils moyens
    profiles_df = pd.read_csv('results/cluster_profiles.csv', sep=';')
    
    print(f"\n📊 Profils disponibles : {len(profiles_df)} clusters")
    print(f"\nAperçu :")
    print(profiles_df[['cluster', 'n_samples', 'percentage']].to_string(index=False))
    
    # Exemple : obtenir le profil moyen du cluster 0
    cluster_0_profile = profiles_df[profiles_df['cluster'] == 0].iloc[0]
    
    # Extraire les valeurs moyennes de J-60 à J-0
    mean_cols = [col for col in profiles_df.columns if col.endswith('_mean')]
    mean_values = cluster_0_profile[mean_cols].values
    
    print(f"\n📈 Profil moyen du cluster 0 :")
    print(f"  - Nombre d'échantillons : {cluster_0_profile['n_samples']}")
    print(f"  - Pourcentage : {cluster_0_profile['percentage']:.1f}%")
    print(f"  - To moyen à J-60 : {cluster_0_profile['J-60_mean']:.3f}")
    print(f"  - To moyen à J : {cluster_0_profile['J-0_mean']:.3f}")
    print(f"  - Croissance : {cluster_0_profile['J-0_mean'] - cluster_0_profile['J-60_mean']:.3f}")
    
    return profiles_df


def example_3_use_in_prediction_model():
    """
    Exemple 3 : Utiliser le cluster dans un modèle de prédiction du To.
    
    Cette fonction montre comment intégrer le cluster comme feature 
    dans un modèle de machine learning pour prédire le To final.
    """
    print("\n" + "="*80)
    print("EXEMPLE 3 : Utilisation dans un modèle de prédiction")
    print("="*80)
    
    # Charger le modèle de clustering
    clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
    clustering.load_model('results/clustering_model.pkl')
    clustering.load_data()
    clustering.prepare_booking_curves()
    
    # Charger les résultats de clustering
    results_df = pd.read_csv('results/clustering_results.csv', sep=';')
    
    print(f"\n📊 Données disponibles : {len(results_df)} dates de séjour avec leur cluster")
    
    # Créer un DataFrame pour l'entraînement d'un modèle de prédiction
    # On va créer des features basées sur :
    # 1. Le cluster assigné
    # 2. Les valeurs de To à différents moments (J-60, J-45, J-30, J-15, J-7)
    # 3. La cible : To à J
    
    feature_cols = ['cluster', 'J-60', 'J-45', 'J-30', 'J-15', 'J-7']
    target_col = 'J-0'
    
    # Ajouter des features dérivées
    results_df['growth_60_to_30'] = results_df['J-30'] - results_df['J-60']
    results_df['growth_30_to_15'] = results_df['J-15'] - results_df['J-30']
    results_df['growth_15_to_7'] = results_df['J-7'] - results_df['J-15']
    
    feature_cols_extended = feature_cols + ['growth_60_to_30', 'growth_30_to_15', 'growth_15_to_7']
    
    # Créer un one-hot encoding du cluster
    cluster_dummies = pd.get_dummies(results_df['cluster'], prefix='cluster')
    
    # Combiner les features
    X = pd.concat([
        results_df[['J-60', 'J-45', 'J-30', 'J-15', 'J-7', 
                    'growth_60_to_30', 'growth_30_to_15', 'growth_15_to_7']],
        cluster_dummies
    ], axis=1)
    
    y = results_df[target_col]
    
    print(f"\n✓ Features créées pour le modèle de prédiction :")
    print(f"  - Nombre de features : {X.shape[1]}")
    print(f"  - Features temporelles : J-60, J-45, J-30, J-15, J-7")
    print(f"  - Features de croissance : growth_60_to_30, growth_30_to_15, growth_15_to_7")
    print(f"  - Features de cluster : {list(cluster_dummies.columns)}")
    print(f"  - Target : {target_col}")
    print(f"  - Nombre d'échantillons : {len(X)}")
    
    # Exemple : statistiques par cluster
    print(f"\n📊 Statistiques du To final (J-0) par cluster :")
    for cluster_id in sorted(results_df['cluster'].unique()):
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        print(f"  - Cluster {cluster_id}: "
              f"n={len(cluster_data)}, "
              f"To moyen={cluster_data['J-0'].mean():.3f}, "
              f"écart-type={cluster_data['J-0'].std():.3f}")
    
    return X, y, results_df


def example_4_predict_future_date():
    """
    Exemple 4 : Prédire le cluster pour une date future (cas réel d'utilisation).
    
    Scénario : Nous sommes à J-15 d'une date de séjour. On a les données 
    de J-60 à J-15, et on veut prédire le To final à J.
    """
    print("\n" + "="*80)
    print("EXEMPLE 4 : Prédiction pour une date future (cas réel)")
    print("="*80)
    
    # Charger le modèle
    clustering = HotelBookingClustering(csv_path='data/Indicateurs.csv', days_before=60)
    clustering.load_model('results/clustering_model.pkl')
    clustering.load_data()
    clustering.prepare_booking_curves()
    
    # Charger les profils
    profiles_df = pd.read_csv('results/cluster_profiles.csv', sep=';')
    
    # Simuler une courbe partielle (on a seulement jusqu'à J-15)
    print("\n📅 Scénario : Nous sommes à J-15 d'une date de séjour")
    print("   Objectif : Prédire le To final à J")
    
    # Exemple de courbe réelle jusqu'à J-15
    partial_curve = {}
    for i in range(60, 14, -1):  # De J-60 à J-15
        # Simuler une montée progressive
        partial_curve[f'J-{i}'] = 0.10 + (60 - i) * 0.01
    
    print(f"\n  - Données disponibles : J-60 à J-15 ({len(partial_curve)} jours)")
    print(f"  - To actuel à J-15 : {partial_curve['J-15']:.3f}")
    
    # Prédire le cluster
    prediction = clustering.predict_cluster(partial_curve)
    
    print(f"\n✓ Cluster prédit : {prediction['cluster']}")
    print(f"  - Confiance : {prediction['confidence']:.3f}")
    
    # Obtenir le profil moyen du cluster pour estimer le To final
    cluster_profile = profiles_df[profiles_df['cluster'] == prediction['cluster']].iloc[0]
    predicted_to_final = cluster_profile['J-0_mean']
    predicted_to_std = cluster_profile['J-0_std']
    
    print(f"\n📈 Prédiction du To final basée sur le cluster {prediction['cluster']} :")
    print(f"  - To final prédit (moyenne du cluster) : {predicted_to_final:.3f} ({predicted_to_final*100:.1f}%)")
    print(f"  - Écart-type : {predicted_to_std:.3f}")
    print(f"  - Intervalle de confiance (±1 std) : "
          f"[{predicted_to_final - predicted_to_std:.3f}, "
          f"{predicted_to_final + predicted_to_std:.3f}]")
    
    # Calculer la croissance attendue
    current_to = partial_curve['J-15']
    expected_growth = predicted_to_final - current_to
    
    print(f"\n  - Croissance attendue (J-15 → J) : {expected_growth:.3f} ({expected_growth*100:.1f} points de %)")
    
    return prediction, predicted_to_final


def main():
    """
    Exécute tous les exemples.
    """
    print("="*80)
    print("EXEMPLES D'UTILISATION DU CLUSTERING POUR LA PRÉDICTION")
    print("="*80)
    
    # Exemple 1 : Prédiction avec données partielles
    try:
        prediction, profile = example_1_predict_with_partial_data()
    except Exception as e:
        print(f"\n⚠️ Exemple 1 échoué : {e}")
    
    # Exemple 2 : Charger les profils depuis CSV
    try:
        profiles_df = example_2_load_profiles_from_csv()
    except Exception as e:
        print(f"\n⚠️ Exemple 2 échoué : {e}")
    
    # Exemple 3 : Utilisation dans un modèle de prédiction
    try:
        X, y, results_df = example_3_use_in_prediction_model()
    except Exception as e:
        print(f"\n⚠️ Exemple 3 échoué : {e}")
    
    # Exemple 4 : Prédiction pour une date future
    try:
        prediction, predicted_to = example_4_predict_future_date()
    except Exception as e:
        print(f"\n⚠️ Exemple 4 échoué : {e}")
    
    print("\n" + "="*80)
    print("✅ EXEMPLES TERMINÉS")
    print("="*80)
    print("\n💡 INTÉGRATION DANS VOTRE MODÈLE DE PRÉDICTION :")
    print("\n1. Dans votre script d'entraînement, chargez le modèle de clustering :")
    print("   from prediction_cluster import HotelBookingClustering")
    print("   clustering = HotelBookingClustering(...)")
    print("   clustering.load_model('results/clustering_model.pkl')")
    print("   clustering.load_data()")
    print("   clustering.prepare_booking_curves()")
    print("\n2. Pour chaque date de séjour, prédisez le cluster :")
    print("   prediction = clustering.predict_cluster(partial_curve)")
    print("   cluster_id = prediction['cluster']")
    print("\n3. Utilisez le cluster comme feature dans votre modèle ML :")
    print("   - Soit en one-hot encoding : cluster_0, cluster_1, ..., cluster_K")
    print("   - Soit avec le profil moyen du cluster : J-60_mean, ..., J-0_mean")
    print("\n4. Entraînez votre modèle avec ces features supplémentaires :")
    print("   - Features : [J-60, J-45, ..., cluster_id, ...]")
    print("   - Target : To final (J-0)")
    print()


if __name__ == "__main__":
    main()

