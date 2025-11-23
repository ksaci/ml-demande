"""
Exemple d'utilisation du modèle PredictTO entraîné pour faire des prédictions.

Ce script montre comment :
1. Charger un modèle sauvegardé
2. Préparer de nouvelles données
3. Faire des prédictions

Usage:
    python predictTo_predict_example.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def load_model_artifacts(model_dir: str = "../results/models"):
    """
    Charge le modèle, le scaler et la liste des features.
    
    Args:
        model_dir: Répertoire contenant les artefacts du modèle
        
    Returns:
        Tuple (model, scaler, feature_cols)
    """
    model_path = Path(model_dir) / "xgb_to_predictor.joblib"
    scaler_path = Path(model_dir) / "xgb_scaler.joblib"
    features_path = Path(model_dir) / "feature_columns.txt"
    
    # Charger le modèle
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé depuis {model_path}")
    
    # Charger le scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler chargé depuis {scaler_path}")
    
    # Charger la liste des features
    with open(features_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    print(f"✅ Features chargées ({len(feature_cols)} colonnes)")
    
    return model, scaler, feature_cols


def compute_pm_features(pm_series):
    """
    Calcule les features compressées à partir d'une série de PM.
    
    Args:
        pm_series: Liste ou array des valeurs PM
        
    Returns:
        Dictionnaire des features calculées
    """
    s = pd.Series(pm_series)
    s = pd.to_numeric(s, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    
    if s.isna().all():
        return {
            "pm_mean": 0.0,
            "pm_slope": 0.0,
            "pm_volatility": 0.0,
            "pm_diff_sum": 0.0,
            "pm_change_ratio": 0.0,
            "pm_last_jump": 0.0,
            "pm_trend_changes": 0,
        }
    
    valid = s.dropna()
    if len(valid) < 2:
        v = float(valid.iloc[0]) if len(valid) == 1 else 0.0
        return {
            "pm_mean": v,
            "pm_slope": 0.0,
            "pm_volatility": 0.0,
            "pm_diff_sum": 0.0,
            "pm_change_ratio": 0.0,
            "pm_last_jump": 0.0,
            "pm_trend_changes": 0,
        }
    
    arr = valid.to_numpy()
    
    pm_mean = float(arr.mean())
    pm_volatility = float(arr.std())
    pm_diff_sum = float(np.sum(np.abs(np.diff(arr))))
    
    x = np.arange(len(arr), dtype=float)
    pm_slope = float(np.polyfit(x, arr, 1)[0])
    
    first = arr[0]
    last = arr[-1]
    pm_change_ratio = float((last - first) / first) if first != 0 else 0.0
    
    if len(arr) >= 6:
        pm_last_jump = float(last - arr[-6])
    else:
        pm_last_jump = float(last - first)
    
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    pm_trend_changes = int(np.sum(np.diff(signs) != 0))
    
    return {
        "pm_mean": pm_mean,
        "pm_slope": pm_slope,
        "pm_volatility": pm_volatility,
        "pm_diff_sum": pm_diff_sum,
        "pm_change_ratio": pm_change_ratio,
        "pm_last_jump": pm_last_jump,
        "pm_trend_changes": pm_trend_changes,
    }


def predict_to(model, scaler, feature_cols, to_series, pm_series, cluster, month, dayofweek, nb_observations=53):
    """
    Fait une prédiction du TO final.
    
    Args:
        model: Modèle XGBoost chargé
        scaler: StandardScaler chargé
        feature_cols: Liste des noms de features
        to_series: Liste des TO de J-60 à J-8 (53 valeurs)
        pm_series: Liste des PM de J-60 à J-8 (53 valeurs)
        cluster: Numéro de cluster (0-N)
        month: Mois du séjour (1-12)
        dayofweek: Jour de la semaine (0-6)
        nb_observations: Nombre d'observations (par défaut 53)
        
    Returns:
        Prédiction du TO final (float entre 0 et 1)
    """
    # Créer le dictionnaire des features
    row_dict = {}
    
    # 1. Ajouter les TO J-60 → J-8
    index_val = 60
    for v in to_series:
        row_dict[f"J-{index_val}"] = v
        index_val -= 1
    
    # 2. Calculer et ajouter les features PM
    pm_feats = compute_pm_features(pm_series)
    row_dict.update(pm_feats)
    
    # 3. Ajouter les variables contextuelles
    row_dict["nb_observations"] = nb_observations
    row_dict["cluster"] = cluster
    row_dict["month"] = month
    row_dict["dayofweek"] = dayofweek
    
    # 4. Créer le DataFrame avec les colonnes dans le bon ordre
    row_df = pd.DataFrame([row_dict], columns=feature_cols)
    
    # 5. Normaliser
    row_scaled = scaler.transform(row_df)
    
    # 6. Prédire
    pred = model.predict(row_scaled)[0]
    
    return float(pred)


def main():
    """
    Exemple d'utilisation du modèle pour faire une prédiction.
    """
    print("=" * 60)
    print("EXEMPLE DE PRÉDICTION AVEC MODÈLE PREDICTTO")
    print("=" * 60)
    
    # 1. Charger les artefacts du modèle
    print("\n📦 Chargement du modèle...")
    model, scaler, feature_cols = load_model_artifacts()
    
    # 2. Préparer des données d'exemple
    print("\n📊 Préparation des données d'exemple...")
    
    # Série TO exemple (J-60 à J-8)
    to_series = [
        0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10, 0.11,
        0.12, 0.13, 0.15, 0.16, 0.18, 0.20, 0.22, 0.23, 0.25, 0.26,
        0.28, 0.30, 0.31, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.42,
        0.43, 0.45, 0.46, 0.47, 0.48, 0.50, 0.51, 0.52, 0.53, 0.55,
        0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.63, 0.64, 0.65, 0.66,
        0.67, 0.68, 0.69  # J-8
    ]
    
    # Série PM exemple (J-60 à J-8)
    pm_series = [
        120, 121, 120, 119, 118, 118, 119, 120, 121, 121,
        122, 123, 123, 124, 125, 124, 123, 123, 124, 124,
        125, 125, 124, 123, 122, 122, 121, 121, 122, 123,
        124, 125, 125, 126, 126, 127, 126, 125, 125, 126,
        127, 127, 126, 125, 124, 124, 123, 122, 122, 123,
        124, 125, 125   # J-8
    ]
    
    # Contexte
    cluster = 3
    month = 8  # Août
    dayofweek = 4  # Vendredi
    
    print(f"   TO à J-8: {to_series[-1]:.2f}")
    print(f"   PM à J-8: {pm_series[-1]:.2f}")
    print(f"   Cluster: {cluster}")
    print(f"   Mois: {month}")
    print(f"   Jour: {dayofweek} (0=Lundi, 6=Dimanche)")
    
    # 3. Faire la prédiction
    print("\n🔮 Prédiction en cours...")
    predicted_to = predict_to(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        to_series=to_series,
        pm_series=pm_series,
        cluster=cluster,
        month=month,
        dayofweek=dayofweek
    )
    
    print("\n" + "=" * 60)
    print(f"✅ PRÉDICTION : TO final = {predicted_to:.4f} ({predicted_to*100:.2f}%)")
    print("=" * 60)
    
    # 4. Analyser le résultat
    current_to = to_series[-1]
    evolution = predicted_to - current_to
    evolution_pct = (evolution / current_to * 100) if current_to > 0 else 0
    
    print(f"\n📈 Analyse:")
    print(f"   TO actuel (J-8): {current_to:.4f} ({current_to*100:.2f}%)")
    print(f"   TO prédit (J-0): {predicted_to:.4f} ({predicted_to*100:.2f}%)")
    print(f"   Évolution: {evolution:+.4f} ({evolution_pct:+.2f}%)")
    
    if evolution > 0:
        print("   📊 Tendance: Montée attendue")
    elif evolution < 0:
        print("   📉 Tendance: Baisse attendue")
    else:
        print("   ➡️  Tendance: Stable")


if __name__ == "__main__":
    main()

