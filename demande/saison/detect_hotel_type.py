#!/usr/bin/env python3
"""
Script de détection automatique du type d'hôtel
Basé sur l'analyse de saisonnalité, ratio semaine/weekend, lead-time, etc.

Usage:
    python detect_hotel_type.py <hotCode>
    
Exemple:
    python detect_hotel_type.py ASB
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path


def calculate_hotel_profile(df):
    """
    Calcule le profil unique d'un hôtel basé sur ses données historiques.
    
    Args:
        df: DataFrame avec colonnes Date, To (taux occupation), Pm (prix moyen), Ant (anticipation), Sel (sélectivité)
    
    Returns:
        dict: Dictionnaire de features décrivant le profil de l'hôtel
    """
    # Agrégations mensuelles
    TO_monthly = df.groupby('Month')['To'].mean()
    PM_monthly = df.groupby('Month')['Pm'].mean()
    
    # Éviter la division par zéro
    to_weekday = df[df.Weekday < 5]['To'].mean()
    to_weekend = df[df.Weekday >= 5]['To'].mean()
    weekend_ratio = to_weekend / to_weekday if to_weekday > 0 else 1.0
    
    profile = {
        # Semaine vs weekend
        "TO_weekday": to_weekday,
        "TO_weekend": to_weekend,
        "weekend_ratio": weekend_ratio,
        
        # Pics été (Juin → Août)
        "summer_peak": TO_monthly.loc[6:8].mean(),
        
        # Pics hiver (Décembre → Mars)
        "winter_peak": pd.concat([TO_monthly.loc[12:12], TO_monthly.loc[1:3]]).mean(),
        
        # Amplitude saisonnière
        "seasonality_amplitude": TO_monthly.max() - TO_monthly.min(),
        
        # Variation prix
        "pm_seasonality": PM_monthly.max() - PM_monthly.min(),
        
        # Lead-time moyen
        "avg_lead_time": df['Ant'].mean(),
        
        # Stabilité du pickup
        "pickup_speed": df['Sel'].mean() if "Sel" in df.columns else None,
    }
    
    return profile


def detect_hotel_type(p):
    """
    Classifieur rule-based pour détecter le type d'hôtel.
    Basé sur l'expertise RMS (Revenue Management System).
    
    Args:
        p: dict - profil de l'hôtel (features calculées)
    
    Returns:
        str: Type d'hôtel détecté
    """
    
    # Calculs préliminaires
    ratio_summer_winter = p["summer_peak"] / p["winter_peak"] if p["winter_peak"] > 0.01 else 999
    ratio_winter_summer = p["winter_peak"] / p["summer_peak"] if p["summer_peak"] > 0.01 else 999
    
    # ========== 1. HÔTEL MER / LOISIRS ==========
    # Critères : fort été, faible hiver, weekend fort, forte saisonnalité
    if (
        ratio_summer_winter > 2.0                  # été beaucoup plus fort
        and p["seasonality_amplitude"] > 0.25      # forte saisonnalité
        and p["weekend_ratio"] >= 0.92             # weekend au moins aussi bon
        and p["summer_peak"] > 0.15                # été significatif
    ):
        return "Hôtel Mer / Loisirs"
    
    # ========== 2. HÔTEL LOISIRS ÉTÉ ==========
    # Critères : pic été marqué mais moins extrême que Mer
    if (
        ratio_summer_winter > 1.6                  # été nettement plus fort
        and p["seasonality_amplitude"] > 0.15      # saisonnalité marquée
        and p["weekend_ratio"] >= 0.95             # weekend bon
        and p["summer_peak"] > 0.10                # été significatif
    ):
        return "Hôtel Loisirs Été"
    
    # ========== 3. HÔTEL MONTAGNE / SKI ==========
    # Critères : fort hiver, faible été, forte saisonnalité
    if (
        ratio_winter_summer > 1.5                  # hiver >> été
        and p["seasonality_amplitude"] > 0.20
        and p["winter_peak"] > 0.08                # hiver significatif
    ):
        return "Hôtel Montagne / Ski"
    
    # ========== 4. URBAIN / BUSINESS STRICT ==========
    # Critères : semaine >> weekend, été creux, faible saisonnalité
    if (
        p["weekend_ratio"] < 0.85                  # weekend clairement plus faible
        and p["TO_weekday"] > p["TO_weekend"] * 1.15
        and p["seasonality_amplitude"] < 0.18      # faible saisonnalité
    ):
        return "Hôtel Urbain / Business"
    
    # ========== 5. ROUTE / ÉCONOMIQUE ==========
    # Critères : très faible saisonnalité, prix stables
    if (
        p["seasonality_amplitude"] < 0.10           # très peu de saisonnalité
        and p["pm_seasonality"] < 25                # prix très stables
        and 0.80 < p["weekend_ratio"] < 1.05        # weekend proche de la semaine
    ):
        return "Hôtel Routier / Économique"
    
    # ========== 6. LOISIRS GÉNÉRAL ==========
    # Critères : weekend fort ET saisonnalité modérée
    if (
        p["weekend_ratio"] > 1.03                   # weekend meilleur
        and p["seasonality_amplitude"] > 0.15       # saisonnalité notable
    ):
        return "Hôtel Loisirs Général"
    
    # ========== 7. URBAIN AVEC SAISONNALITÉ ==========
    # Critères : semaine meilleure avec de la saisonnalité (congrès, foires)
    if (
        p["weekend_ratio"] < 0.98                   # semaine meilleure
        and p["seasonality_amplitude"] > 0.15
        and p["TO_weekday"] > 0.30                  # volume significatif
    ):
        return "Hôtel Urbain avec Saisonnalité"
    
    # ========== 8. MIXTE ÉQUILIBRÉ ==========
    # Critères : profil équilibré sans dominante claire
    if (
        0.92 < p["weekend_ratio"] < 1.05           # semaine ≈ weekend
        and p["seasonality_amplitude"] < 0.22      # saisonnalité modérée
    ):
        return "Hôtel Mixte Équilibré"
    
    # ========== 9. PAR DÉFAUT ==========
    return "Hôtel Indéterminé (profil mixte)"


def load_hotel_data(hotCode, data_dir='../data'):
    """
    Charge les données d'un hôtel depuis le fichier CSV.
    
    Args:
        hotCode: Code de l'hôtel (ex: 'ASB')
        data_dir: Répertoire contenant les données
    
    Returns:
        DataFrame avec les colonnes nécessaires
    """
    filepath = Path(data_dir) / hotCode / 'Indicateurs.csv'
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    
    df = pd.read_csv(filepath, sep=';')
    df.fillna(0, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    
    return df


def print_results(hotel_type, profile, hotCode):
    """
    Affiche les résultats de la détection de manière formatée.
    """
    print("=" * 60)
    print(f"ANALYSE DU PROFIL HÔTEL : {hotCode}")
    print("=" * 60)
    print(f"\n🏨 Type détecté : {hotel_type}")
    print("\n" + "-" * 60)
    print("Indicateurs utilisés pour l'analyse :")
    print("-" * 60)
    
    indicators = [
        ("TO Semaine", profile["TO_weekday"], "{:.2%}"),
        ("TO Weekend", profile["TO_weekend"], "{:.2%}"),
        ("Ratio Weekend/Semaine", profile["weekend_ratio"], "{:.2f}"),
        ("Pic Été (Jun-Aoû)", profile["summer_peak"], "{:.2%}"),
        ("Pic Hiver (Déc-Mar)", profile["winter_peak"], "{:.2%}"),
        ("Amplitude Saisonnière", profile["seasonality_amplitude"], "{:.2%}"),
        ("Variation Prix (PM)", profile["pm_seasonality"], "{:.2f} €"),
        ("Lead-time Moyen", profile["avg_lead_time"], "{:.1f} jours"),
        ("Vitesse Pickup (Sel)", profile["pickup_speed"], "{:.2f}"),
    ]
    
    for label, value, fmt in indicators:
        if value is not None:
            print(f"  • {label:<25} : {fmt.format(value)}")
        else:
            print(f"  • {label:<25} : N/A")
    
    print("=" * 60)


def main():
    """
    Point d'entrée principal du script.
    """
    # Vérification des arguments
    if len(sys.argv) < 2:
        print("❌ Erreur : Code hôtel manquant")
        print(f"\nUsage: python {sys.argv[0]} <hotCode>")
        print(f"Exemple: python {sys.argv[0]} ASB")
        sys.exit(1)
    
    hotCode = sys.argv[1]
    
    try:
        # Chargement des données
        print(f"📊 Chargement des données pour l'hôtel {hotCode}...")
        df = load_hotel_data(hotCode)
        
        # Calcul du profil
        print("🔍 Calcul du profil de l'hôtel...")
        profile = calculate_hotel_profile(df)
        
        # Détection du type
        print("🎯 Détection du type d'hôtel...\n")
        hotel_type = detect_hotel_type(profile)
        
        # Affichage des résultats
        print_results(hotel_type, profile, hotCode)
        
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

