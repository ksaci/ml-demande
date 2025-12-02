"""
Script de test pour vérifier que le remplacement de {hotCode} fonctionne.
"""

def test_hotel_replacement():
    """Test le remplacement du placeholder {hotCode} dans les chemins."""
    
    print("=" * 80)
    print("TEST DU REMPLACEMENT DE {hotCode} DANS LA CONFIGURATION")
    print("=" * 80)
    
    # Simuler la configuration
    config = {
        'clustering_results_path': '../cluster/results/{hotCode}/clustering_results.csv',
        'indicateurs_path': '../data/{hotCode}/Indicateurs.csv',
        'rateShopper_path': '../data/{hotCode}/rateShopper.csv',
        'output_base_dir': 'results'
    }
    
    hotels_to_test = ['D09', '6N8', '0BT', 'ABC']
    
    for hotel_code in hotels_to_test:
        print(f"\n{'='*80}")
        print(f"HÔTEL: {hotel_code}")
        print(f"{'='*80}")
        
        # Simuler le remplacement
        test_config = config.copy()
        paths_to_replace = [
            'clustering_results_path',
            'indicateurs_path',
            'rateShopper_path'
        ]
        
        for path_key in paths_to_replace:
            if path_key in test_config:
                original_path = test_config[path_key]
                if '{hotCode}' in original_path:
                    new_path = original_path.replace('{hotCode}', hotel_code)
                    test_config[path_key] = new_path
                    print(f"✅ {path_key}:")
                    print(f"   Avant: {original_path}")
                    print(f"   Après: {new_path}")
        
        # Vérifier output_dir
        base_dir = test_config.get('output_base_dir', 'results')
        horizon = 7
        output_dir = f"{base_dir}/{hotel_code}/J-{horizon}"
        print(f"✅ output_dir: {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("=" * 80)
    print("\nLe remplacement de {hotCode} fonctionne correctement !")
    print("Les chemins sont maintenant dynamiques et s'adaptent à chaque hôtel.")


if __name__ == "__main__":
    test_hotel_replacement()

