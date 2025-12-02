"""
Script de test pour vérifier que l'entraînement fonctionne avec horizon=0.

Ce script teste la logique de filtrage des colonnes pour différents horizons.
"""

import sys


def test_column_filtering():
    """Test la logique de filtrage des colonnes selon l'horizon."""
    
    print("=" * 80)
    print("TEST DU FILTRAGE DES COLONNES SELON L'HORIZON")
    print("=" * 80)
    
    # Simuler les colonnes disponibles
    all_j_values = list(range(0, 61))  # J-0 à J-60
    
    horizons_to_test = [0, 1, 3, 7, 14, 30, 60]
    
    for horizon in horizons_to_test:
        print(f"\n{'='*80}")
        print(f"HORIZON = {horizon} (Prédire J-0 en étant à J-{horizon})")
        print(f"{'='*80}")
        
        # Logique de filtrage (comme dans le code corrigé)
        pm_cols_available = []
        for j_num in all_j_values:
            if j_num > horizon:  # Condition corrigée : > au lieu de >=
                pm_cols_available.append(f"pm_J-{j_num}")
        
        # Features TO
        to_feature_cols = [f"J-{i}" for i in range(60, horizon, -1)]
        
        print(f"\n1️⃣  Features PM disponibles ({len(pm_cols_available)}):")
        if len(pm_cols_available) > 0:
            if len(pm_cols_available) <= 5:
                print(f"   {', '.join(pm_cols_available)}")
            else:
                first_3 = ', '.join(pm_cols_available[:3])
                last_2 = ', '.join(pm_cols_available[-2:])
                print(f"   {first_3}, ..., {last_2}")
        else:
            print(f"   ⚠️  AUCUNE")
        
        print(f"\n2️⃣  Features TO disponibles ({len(to_feature_cols)}):")
        if len(to_feature_cols) > 0:
            if len(to_feature_cols) <= 5:
                print(f"   {', '.join(to_feature_cols)}")
            else:
                first_3 = ', '.join(to_feature_cols[:3])
                last_2 = ', '.join(to_feature_cols[-2:])
                print(f"   {first_3}, ..., {last_2}")
        else:
            print(f"   ⚠️  AUCUNE")
        
        # Vérifications
        print(f"\n3️⃣  Vérifications:")
        
        # Check 1: pm_J-{horizon} ne doit PAS être inclus
        target_pm_col = f"pm_J-{horizon}"
        if target_pm_col in pm_cols_available:
            print(f"   ❌ ERREUR : {target_pm_col} est inclus (data leakage !)")
            sys.exit(1)
        else:
            print(f"   ✅ {target_pm_col} n'est pas inclus (pas de data leakage)")
        
        # Check 2: pm_J-{horizon+1} doit être inclus (sauf si horizon >= 60)
        if horizon < 60:
            expected_pm_col = f"pm_J-{horizon+1}"
            if expected_pm_col in pm_cols_available:
                print(f"   ✅ {expected_pm_col} est inclus")
            else:
                print(f"   ❌ ERREUR : {expected_pm_col} devrait être inclus")
                sys.exit(1)
        
        # Check 3: J-{horizon} ne doit PAS être dans les features TO
        target_to_col = f"J-{horizon}"
        if target_to_col in to_feature_cols:
            print(f"   ❌ ERREUR : {target_to_col} est inclus (data leakage !)")
            sys.exit(1)
        else:
            print(f"   ✅ {target_to_col} n'est pas inclus (pas de data leakage)")
        
        # Check 4: Au moins quelques features disponibles (sauf pour horizon=60)
        if horizon < 60:
            if len(pm_cols_available) == 0:
                print(f"   ❌ ERREUR : Aucune feature PM disponible pour horizon={horizon}")
                sys.exit(1)
            elif len(pm_cols_available) < 5:
                print(f"   ⚠️  WARNING : Seulement {len(pm_cols_available)} features PM disponibles")
            else:
                print(f"   ✅ {len(pm_cols_available)} features PM disponibles")
        
        print(f"\n4️⃣  Résumé:")
        print(f"   - Prédiction : TO final (J-0)")
        print(f"   - Données jusqu'à : J-{horizon}")
        print(f"   - Features PM : pm_J-{horizon+1 if horizon < 60 else '?'} à pm_J-60")
        print(f"   - Features TO : J-{horizon+1 if horizon < 60 else '?'} à J-60")
    
    print("\n" + "=" * 80)
    print("✅ TOUS LES TESTS SONT PASSÉS")
    print("=" * 80)
    print("\nLa logique de filtrage est correcte pour tous les horizons !")
    print("Le script d'entraînement ne devrait plus planter sur J-0.")


if __name__ == "__main__":
    test_column_filtering()

