"""
Script d'entraînement du modèle XGBoost pour la prédiction du taux d'occupation (TO).

Ce script permet de :
1. Charger les données de clustering et les indicateurs PM/RevPAR
2. Préparer les features (séries temporelles TO, features PM compressées)
3. Entraîner un modèle XGBoost pour prédire TO à J+7
4. Évaluer les performances (MAE, R², graphiques)
5. Sauvegarder le modèle dans Azure Blob Storage

Auteur: Équipe Data Science
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import holidays

# Import optionnel pour vacances scolaires
try:
    from vacances_scolaires_france import SchoolHolidayDates
    SCHOOL_HOLIDAYS_AVAILABLE = True
except ImportError:
    SCHOOL_HOLIDAYS_AVAILABLE = False

# Import optionnel pour YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import optionnel pour dotenv (chargement du fichier .env)
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    # Charger le fichier .env s'il existe
    load_dotenv()
except ImportError:
    DOTENV_AVAILABLE = False

# Configuration du logging
# Classe personnalisée pour gérer l'encodage UTF-8 sous Windows
class UTF8StreamHandler(logging.StreamHandler):
    """StreamHandler qui force l'encodage UTF-8 pour éviter les erreurs avec les emojis."""
    def __init__(self, stream=None):
        super().__init__(stream)
        # Tenter de reconfigurer stdout en UTF-8 si possible (Python 3.7+)
        if stream is None or stream is sys.stdout:
            if hasattr(sys.stdout, 'reconfigure'):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                except (AttributeError, ValueError):
                    pass  # Si la reconfiguration échoue, on continue
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Encoder en UTF-8 avec gestion d'erreurs
            if hasattr(stream, 'buffer'):
                stream.buffer.write(msg.encode('utf-8', errors='replace'))
                stream.buffer.write(self.terminator.encode('utf-8'))
                stream.buffer.flush()
            else:
                stream.write(msg + self.terminator)
                stream.flush()
        except UnicodeEncodeError:
            # En cas d'erreur, remplacer les caractères problématiques
            try:
                msg = self.format(record)
                msg_safe = msg.encode('ascii', errors='replace').decode('ascii')
                stream.write(msg_safe + self.terminator)
                stream.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictTo_training.log', encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Désactiver les logs verbeux du SDK Azure
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.core').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)


class XGBoostOccupancyPredictor:
    """
    Classe principale pour l'entraînement du modèle XGBoost de prédiction du TO.
    """
    
    def __init__(self, config: Dict[str, Any], hotel_code: str = None):
        """
        Initialise le prédicteur avec la configuration.
        
        Args:
            config: Dictionnaire de configuration contenant les paramètres du modèle
            hotel_code: Code de l'hôtel (optionnel, pour entraînement par hôtel)
        """
        self.config = config
        self.hotel_code = hotel_code
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.results = {}
        
        # Remplacer {hotel} dans les chemins de configuration si hotel_code est fourni
        if hotel_code:
            self._replace_hotel_placeholder()
        
        logger.info("Initialisation du XGBoostOccupancyPredictor")
        if hotel_code:
            logger.info(f"Mode: Entraînement pour l'hôtel {hotel_code}")
        else:
            logger.info(f"Mode: Entraînement global (tous les hôtels)")
        logger.info(f"Configuration après substitution: {config}")
    
    def _replace_hotel_placeholder(self):
        """
        Remplace le placeholder {hotCode} dans les chemins de configuration par le code d'hôtel réel.
        """
        if not self.hotel_code:
            return
        
        # Chemins à remplacer
        paths_to_replace = [
            'clustering_results_path',
            'indicateurs_path',
            'rateShopper_path'
        ]
        
        for path_key in paths_to_replace:
            if path_key in self.config:
                original_path = self.config[path_key]
                if '{hotCode}' in original_path:
                    new_path = original_path.replace('{hotCode}', self.hotel_code)
                    self.config[path_key] = new_path
                    logger.debug(f"Remplacement {path_key}: {original_path} → {new_path}")
    
    def _get_output_dir(self) -> str:
        """
        Construit le répertoire de sortie selon la structure hotel/horizon.
        
        Returns:
            Chemin du répertoire de sortie (ex: results/D09/J-7 ou results/ALL/J-7)
        """
        base_dir = self.config.get('output_base_dir', 'results')
        horizon = self.config.get('prediction_horizon', 7)
        
        # Si un hotel_code est spécifié, l'utiliser, sinon "ALL"
        hotel_folder = self.hotel_code if self.hotel_code else "ALL"
        
        # Construire le chemin : base_dir/hotel/J-horizon
        output_dir = os.path.join(base_dir, hotel_folder, f"J-{horizon}")
        
        return output_dir
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge les données de clustering et les indicateurs.
        Si un hotel_code est spécifié, charge depuis cluster/results/{hotel_code}/clustering_results.csv
        Sinon, utilise le chemin de la configuration.
        
        Returns:
            Tuple contenant (clusters_df, indicateurs_df)
        """
        logger.info("Chargement des données...")
        
        try:
            # Déterminer le chemin du fichier de clustering
            if self.hotel_code:
                # Construire le chemin dynamiquement depuis le dossier cluster/results/{hotel}/
                clustering_base_dir = self.config.get('clustering_base_dir', '../cluster/results')
                clustering_path = os.path.join(clustering_base_dir, self.hotel_code, 'clustering_results.csv')
                logger.info(f"Chargement des clusters depuis: {clustering_path}")
            else:
                # Utiliser le chemin de la configuration (mode global)
                clustering_path = self.config['clustering_results_path']
                logger.info(f"Chargement des clusters depuis la config: {clustering_path}")
            
            # Charger les résultats de clustering
            clusters = pd.read_csv(clustering_path, sep=';')
            logger.info(f"✅ Clusters chargés: {clusters.shape}")
            
            # Charger les indicateurs
            indicateurs = pd.read_csv(
                self.config['indicateurs_path'], 
                sep=';'
            )
            logger.info(f"✅ Indicateurs chargés: {indicateurs.shape}")
            
            return clusters, indicateurs
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des données: {e}")
            raise
    
    def load_competitor_prices(self, rateShopper_path: str) -> pd.DataFrame:
        """
        Charge les prix des concurrents depuis le fichier rateShopper.
        
        Args:
            rateShopper_path: Chemin vers le fichier rateShopper.csv
            
        Returns:
            DataFrame avec les prix des concurrents
        """
        try:
            logger.info(f"Chargement des prix concurrents depuis {rateShopper_path}...")
            df_comp = pd.read_csv(rateShopper_path, sep=';')
            
            # Renommer les colonnes pour cohérence
            df_comp = df_comp.rename(columns={
                'HotCode': 'hotCode',
                'Date': 'Date',
                'DateImport': 'ObsDate'
            })
            
            logger.info(f"Prix concurrents chargés: {df_comp.shape}")
            return df_comp
            
        except FileNotFoundError:
            logger.warning(f"⚠️  Fichier {rateShopper_path} non trouvé. Les features concurrentes ne seront pas ajoutées.")
            return None
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors du chargement des prix concurrents: {e}")
            return None
    
    def prepare_competitor_features(self, df: pd.DataFrame, df_comp: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features à partir des prix médians des concurrents.
        
        Args:
            df: DataFrame principal avec stay_date et hotCode
            df_comp: DataFrame des prix concurrents
            
        Returns:
            DataFrame enrichi avec les features concurrentes (prix médian uniquement)
        """
        if df_comp is None:
            logger.info("Pas de données concurrentes disponibles")
            return df
        
        logger.info("Préparation des features concurrentes (prix médian)...")
        
        # Convertir les dates et retirer les informations de timezone
        df_comp["Date"] = pd.to_datetime(df_comp["Date"]).dt.tz_localize(None)
        df_comp["ObsDate"] = pd.to_datetime(df_comp["ObsDate"]).dt.tz_localize(None)
        
        # Calculer la distance J-x
        df_comp["days_before"] = (df_comp["Date"] - df_comp["ObsDate"]).dt.days
        
        # Garder uniquement J-0 → J-60
        df_comp = df_comp[
            (df_comp["days_before"] >= 0) & 
            (df_comp["days_before"] <= 60)
        ]
        
        # Pivoter les données pour le prix médian uniquement
        pivot = df_comp.pivot_table(
            index=["hotCode", "Date"],
            columns="days_before",
            values="CompPrixMedian",
            aggfunc="last"
        )
        
        pivot.columns = [f"CompPrixMedian_J-{col}" for col in pivot.columns]
        pivot = pivot.reset_index()
        
        # Récupérer les colonnes de séries temporelles (uniquement J-(horizon+1) à J-60)
        horizon = self.config['prediction_horizon']
        comp_cols_all = [c for c in pivot.columns if c.startswith("CompPrixMedian_J-")]
        comp_cols_available = []
        for col in comp_cols_all:
            # Extraire le numéro de J-X
            j_num = int(col.split("J-")[1])
            if j_num > horizon:  # Seulement les données APRÈS J-horizon (pas de data leakage)
                comp_cols_available.append(col)
        
        comp_cols_available = sorted(comp_cols_available, key=lambda x: int(x.split("J-")[1]))
        
        logger.info(f"   Features concurrentes calculées sur J-{horizon+1} à J-60")
        
        if len(comp_cols_available) == 0:
            logger.warning(f"⚠️  Aucune colonne Comp disponible pour horizon={horizon}")
        
        # Convertir en numérique
        pivot[comp_cols_available] = pivot[comp_cols_available].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        
        # Fusionner le pivot Comp avec le DataFrame principal pour avoir PM et Comp alignés
        df = df.merge(
            pivot,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "Date"],
            how="left"
        ).drop(columns=["Date"], errors='ignore')
        
        logger.info("Construction des séries Gap et Elasticity...")
        
        # Construire les colonnes PM correspondantes (même horizon)
        pm_cols_for_gap = []
        for col in comp_cols_available:
            j_num = col.split("J-")[1]
            pm_col = f"pm_J-{j_num}"
            if pm_col in df.columns:
                pm_cols_for_gap.append(pm_col)
        
        # Calculer les features Gap et Elasticity pour chaque ligne
        features_list = []
        for idx, row in df.iterrows():
            # Récupérer les séries PM et Comp
            pm_series = row[pm_cols_for_gap].values
            comp_series = row[comp_cols_available].values
            
            # Convertir en arrays numériques
            pm_arr = pd.to_numeric(pd.Series(pm_series), errors='coerce').replace([np.inf, -np.inf], np.nan).values
            comp_arr = pd.to_numeric(pd.Series(comp_series), errors='coerce').replace([np.inf, -np.inf], np.nan).values
            
            # 1. Gap series = PM - Comp
            gap_series = pm_arr - comp_arr
            gap_valid = pd.Series(gap_series).dropna()
            
            if len(gap_valid) >= 1:
                gap_last = float(gap_valid.iloc[0])  # Première valeur = J-horizon
                if len(gap_valid) >= 2:
                    x = np.arange(len(gap_valid))
                    gap_slope = float(np.polyfit(x, gap_valid.values, 1)[0])
                else:
                    gap_slope = 0.0
            else:
                gap_last = 0.0
                gap_slope = 0.0
            
            # 2. Elasticity series = PM / Comp
            elasticity_series = pm_arr / np.where(comp_arr == 0, np.nan, comp_arr)
            elasticity_valid = pd.Series(elasticity_series).dropna()
            
            if len(elasticity_valid) >= 1:
                elasticity_last = float(elasticity_valid.iloc[0])  # Première valeur = J-horizon
                if len(elasticity_valid) >= 2:
                    x = np.arange(len(elasticity_valid))
                    elasticity_slope = float(np.polyfit(x, elasticity_valid.values, 1)[0])
                else:
                    elasticity_slope = 0.0
            else:
                elasticity_last = 1.0  # Valeur neutre si pas de données
                elasticity_slope = 0.0
            
            # Features compressées classiques sur Comp
            comp_series_clean = pd.Series(comp_series)
            feats_comp = self.compute_price_features(comp_series_clean, prefix="comp")
            
            # Ajouter toutes les features
            feats = {
                'hotCode': row['hotCode'],
                'stay_date_key': row['stay_date'],
                'gap_last': gap_last,
                'gap_slope': gap_slope,
                'elasticity_last': elasticity_last,
                'elasticity_slope': elasticity_slope,
                **feats_comp
            }
            features_list.append(feats)
        
        df_all_feats = pd.DataFrame(features_list)
        
        # Fusionner avec le DataFrame principal
        df = df.merge(
            df_all_feats,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "stay_date_key"],
            how="left"
        ).drop(columns=["stay_date_key"], errors='ignore')
        
        # Supprimer les colonnes pivotées CompPrixMedian_J-X pour alléger
        cols_to_drop = [c for c in df.columns if c.startswith("CompPrixMedian_J-")]
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info(f"Features concurrentes + Gap/Elasticity ajoutées (10 features). Shape: {df.shape}")
        
        return df
    
    def compute_price_features(self, price_series_raw: pd.Series, prefix: str = "pm") -> Dict[str, float]:
        """
        Calcule les features compressées à partir d'une série de prix.
        
        Args:
            price_series_raw: Série temporelle des prix
            prefix: Préfixe pour les noms des features (ex: "pm", "comp_min", etc.)
            
        Returns:
            Dictionnaire des features calculées
        """
        # Conversion en Series pour utiliser les outils pandas
        s = pd.Series(price_series_raw)
        
        # Conversion en numérique (float), tout ce qui n'est pas convertible -> NaN
        s = pd.to_numeric(s, errors='coerce')
        
        # Remplacement des +/-inf éventuels
        s = s.replace([np.inf, -np.inf], np.nan)
        
        # Si tout est NaN -> on renvoie des 0 safe
        if s.isna().all():
            return {
                f"{prefix}_slope": 0.0,
                f"{prefix}_volatility": 0.0,
                f"{prefix}_diff_sum": 0.0,
                f"{prefix}_change_ratio": 0.0,
                f"{prefix}_last_jump": 0.0,
                f"{prefix}_trend_changes": 0,
            }
        
        # Si après interpolation il reste moins de 2 points valides -> pas de pente possible
        valid = s.dropna()
        if len(valid) < 2:
            v = float(valid.iloc[0]) if len(valid) == 1 else 0.0
            return {
                f"{prefix}_slope": 0.0,
                f"{prefix}_volatility": 0.0,
                f"{prefix}_diff_sum": 0.0,
                f"{prefix}_change_ratio": 0.0,
                f"{prefix}_last_jump": 0.0,
                f"{prefix}_trend_changes": 0,
            }
        
        arr = valid.to_numpy()
        
        slope = float(np.polyfit(np.arange(len(arr), dtype=float), arr, 1)[0])
        volatility = float(arr.std())
        diff_sum = float(np.sum(np.abs(np.diff(arr))))
        
        # Ratio global
        first = arr[0]
        last = arr[-1]
        change_ratio = float((last - first) / first) if first != 0 else 0.0
        
        # Variation récente
        if len(arr) >= 6:
            last_jump = float(last - arr[-6])
        else:
            last_jump = float(last - first)
        
        # Changements de direction
        diffs = np.diff(arr)
        signs = np.sign(diffs)
        trend_changes = int(np.sum(np.diff(signs) != 0))
        
        return {
            f"{prefix}_slope": slope,
            f"{prefix}_volatility": volatility,
            f"{prefix}_diff_sum": diff_sum,
            f"{prefix}_change_ratio": change_ratio,
            f"{prefix}_last_jump": last_jump,
            f"{prefix}_trend_changes": trend_changes,
        }
    
    def compute_pm_features(self, pm_series_raw: pd.Series) -> Dict[str, float]:
        """
        Calcule les features compressées à partir d'une série de PM (rétrocompatibilité).
        
        Args:
            pm_series_raw: Série temporelle des prix moyens
            
        Returns:
            Dictionnaire des features calculées (avec préfixe pm et pm_mean)
        """
        features = self.compute_price_features(pm_series_raw, prefix="pm")
        # Ajouter pm_mean pour compatibilité avec le code existant
        s = pd.Series(pm_series_raw)
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
        valid = s.dropna()
        features["pm_mean"] = float(valid.mean()) if len(valid) > 0 else 0.0
        return features
    
    def prepare_data(self, clusters: pd.DataFrame, indicateurs: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            clusters: DataFrame des résultats de clustering
            indicateurs: DataFrame des indicateurs
            
        Returns:
            DataFrame enrichi avec toutes les features
        """
        logger.info("Préparation des données...")
        
        # Convertir les dates
        clusters["stay_date"] = pd.to_datetime(clusters["stay_date"])
        indicateurs["Date"] = pd.to_datetime(indicateurs["Date"])
        indicateurs["ObsDate"] = pd.to_datetime(indicateurs["ObsDate"])
        
        # Calculer la distance J-x pour chaque observation
        indicateurs["days_before"] = (indicateurs["Date"] - indicateurs["ObsDate"]).dt.days
        
        # Garder uniquement les valeurs J-0 → J-60
        indicateurs = indicateurs[
            (indicateurs["days_before"] >= 0) & 
            (indicateurs["days_before"] <= 60)
        ]
        
        # Pivoter les PM
        pm_pivot = indicateurs.pivot_table(
            index=["hotCode", "Date"],
            columns="days_before",
            values="Pm",
            aggfunc="last"
        )
        
        pm_pivot.columns = [f"pm_J-{col}" for col in pm_pivot.columns]
        pm_pivot = pm_pivot.reset_index()
        
        logger.info(f"PM pivot shape: {pm_pivot.shape}")
        
        # Pivoter Ant (Anticipation)
        ant_pivot = indicateurs.pivot_table(
            index=["hotCode", "Date"],
            columns="days_before",
            values="Ant",
            aggfunc="last"
        )
        
        ant_pivot.columns = [f"ant_J-{col}" for col in ant_pivot.columns]
        ant_pivot = ant_pivot.reset_index()
        
        logger.info(f"Ant pivot shape: {ant_pivot.shape}")
        
        # Pivoter Ds (Durée de séjour)
        ds_pivot = indicateurs.pivot_table(
            index=["hotCode", "Date"],
            columns="days_before",
            values="Ds",
            aggfunc="last"
        )
        
        ds_pivot.columns = [f"ds_J-{col}" for col in ds_pivot.columns]
        ds_pivot = ds_pivot.reset_index()
        
        logger.info(f"Ds pivot shape: {ds_pivot.shape}")
        
        # Fusionner avec les clusters
        df = clusters.merge(
            pm_pivot,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "Date"],
            how="left"
        ).drop(columns=["Date"])
        
        # Fusionner Ant
        df = df.merge(
            ant_pivot,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "Date"],
            how="left"
        ).drop(columns=["Date"])
        
        # Fusionner Ds
        df = df.merge(
            ds_pivot,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "Date"],
            how="left"
        ).drop(columns=["Date"])
        
        logger.info(f"DataFrame fusionné: {df.shape}")
        
        # Calculer les features PM compressées (uniquement sur les données disponibles jusqu'à J-horizon)
        horizon = self.config['prediction_horizon']
        
        # Filtrer les colonnes PM : de J-(horizon+1) à J-60 (pas de données futures !)
        # Pour horizon=0, on utilise J-1 à J-60 (pas J-0 car c'est le jour à prédire)
        # Pour horizon=7, on utilise J-7 à J-60
        pm_cols_all = [c for c in df.columns if c.startswith("pm_J-")]
        pm_cols_available = []
        for col in pm_cols_all:
            # Extraire le numéro de J-X
            j_num = int(col.split("J-")[1])
            if j_num > horizon:  # Seulement les données APRÈS J-horizon (pas de data leakage)
                pm_cols_available.append(col)
        
        pm_cols_available = sorted(pm_cols_available, key=lambda x: int(x.split("J-")[1]))
        
        logger.info(f"Calcul des features PM sur données J-{horizon+1} à J-60 (pas de data leakage)")
        logger.info(f"   Colonnes PM utilisées: {len(pm_cols_available)}")
        
        # Vérifier qu'on a au moins quelques colonnes
        if len(pm_cols_available) == 0:
            logger.error(f"❌ Aucune colonne PM disponible pour horizon={horizon}")
            logger.error(f"   Colonnes PM existantes: {[c for c in pm_cols_all[:5]]}")
            raise ValueError(f"Pas de données PM disponibles pour horizon={horizon}")
        
        # Convertir en numérique
        df[pm_cols_available] = df[pm_cols_available].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        
        features_list = []
        for idx, row in df.iterrows():
            pm_series = row[pm_cols_available].values
            feats = self.compute_pm_features(pm_series)
            features_list.append(feats)
        
        df_feats = pd.DataFrame(features_list)
        df = pd.concat([df, df_feats], axis=1)
        
        logger.info(f"Features PM ajoutées: {df.shape}")
        
        # Calculer les features Ant (Anticipation) compressées
        ant_cols_all = [c for c in df.columns if c.startswith("ant_J-")]
        ant_cols_available = []
        for col in ant_cols_all:
            j_num = int(col.split("J-")[1])
            if j_num > horizon:  # Données APRÈS J-horizon (pas de data leakage)
                ant_cols_available.append(col)
        
        ant_cols_available = sorted(ant_cols_available, key=lambda x: int(x.split("J-")[1]))
        
        logger.info(f"Calcul des features Ant (anticipation) sur données J-{horizon+1} à J-60")
        logger.info(f"   Colonnes Ant utilisées: {len(ant_cols_available)}")
        
        if len(ant_cols_available) == 0:
            logger.warning(f"⚠️  Aucune colonne Ant disponible pour horizon={horizon}")
        
        # Convertir en numérique
        df[ant_cols_available] = df[ant_cols_available].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        
        features_list = []
        for idx, row in df.iterrows():
            ant_series = row[ant_cols_available].values
            feats = self.compute_price_features(ant_series, prefix="ant")
            features_list.append(feats)
        
        df_feats_ant = pd.DataFrame(features_list)
        df = pd.concat([df, df_feats_ant], axis=1)
        
        logger.info(f"Features Ant ajoutées: {df.shape}")
        
        # Calculer les features Ds (Durée de séjour) compressées
        ds_cols_all = [c for c in df.columns if c.startswith("ds_J-")]
        ds_cols_available = []
        for col in ds_cols_all:
            j_num = int(col.split("J-")[1])
            if j_num > horizon:  # Données APRÈS J-horizon (pas de data leakage)
                ds_cols_available.append(col)
        
        ds_cols_available = sorted(ds_cols_available, key=lambda x: int(x.split("J-")[1]))
        
        logger.info(f"Calcul des features Ds (durée séjour) sur données J-{horizon+1} à J-60")
        logger.info(f"   Colonnes Ds utilisées: {len(ds_cols_available)}")
        
        if len(ds_cols_available) == 0:
            logger.warning(f"⚠️  Aucune colonne Ds disponible pour horizon={horizon}")
        
        # Convertir en numérique
        df[ds_cols_available] = df[ds_cols_available].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        
        features_list = []
        for idx, row in df.iterrows():
            ds_series = row[ds_cols_available].values
            feats = self.compute_price_features(ds_series, prefix="ds")
            features_list.append(feats)
        
        df_feats_ds = pd.DataFrame(features_list)
        df = pd.concat([df, df_feats_ds], axis=1)
        
        logger.info(f"Features Ds ajoutées: {df.shape}")
        
        # Ajouter les features des prix concurrents
        rateShopper_path = self.config.get('rateShopper_path', '../data/{hotCode}/rateShopper.csv')
        df_comp = self.load_competitor_prices(rateShopper_path)
        if df_comp is not None:
            df = self.prepare_competitor_features(df, df_comp)
        
        # Ajouter les features temporelles
        if not np.issubdtype(df["stay_date"].dtype, np.datetime64):
            df["stay_date"] = pd.to_datetime(df["stay_date"])
        
        df["month"] = df["stay_date"].dt.month
        df["dayofweek"] = df["stay_date"].dt.dayofweek
        
        # Ajouter ToF1 : TO final de l'année dernière (même date l'année précédente)
        logger.info("Ajout de ToF1 (TO final année N-1)...")
        if "J-0" in df.columns:
            # Créer un DataFrame temporaire avec stay_date et J-0
            df_to_last_year = df[["hotCode", "stay_date", "J-0"]].copy()
            df_to_last_year["stay_date_next_year"] = df_to_last_year["stay_date"] + pd.DateOffset(years=1)
            df_to_last_year = df_to_last_year.rename(columns={"J-0": "ToF1"})
            
            # Fusionner pour récupérer le TO de l'année précédente
            df = df.merge(
                df_to_last_year[["hotCode", "stay_date_next_year", "ToF1"]],
                left_on=["hotCode", "stay_date"],
                right_on=["hotCode", "stay_date_next_year"],
                how="left"
            ).drop(columns=["stay_date_next_year"])
            
            # Remplir les valeurs manquantes par 0 (pas de données pour l'année précédente)
            df["ToF1"] = df["ToF1"].fillna(0)
            
            logger.info(f"   ToF1 ajouté : {(df['ToF1'] > 0).sum()} valeurs non-nulles sur {len(df)}")
        else:
            logger.warning("⚠️  Colonne J-0 non trouvée, ToF1 ne peut pas être calculé")
            df["ToF1"] = 0
        
        # Features liées aux jours fériés français
        logger.info("Ajout des features jours fériés...")
        years_needed = range(
            df["stay_date"].dt.year.min(),
            df["stay_date"].dt.year.max() + 1
        )

        fr_holidays = holidays.France(years=list(years_needed))
        # 1. Indicateur jour férié
        df["is_holiday_fr"] = df["stay_date"].apply(
            lambda x: 1 if x.date() in fr_holidays else 0
        )
        
        # 2. Indicateur "pont" (jour avant ou après un férié)
        df["is_bridge_day"] = df["stay_date"].apply(
            lambda d: 1 if (
                ((d - pd.Timedelta(days=1)).date() in fr_holidays) or 
                ((d + pd.Timedelta(days=1)).date() in fr_holidays)
            ) else 0
        )
        
        # 3. Distance au prochain jour férié
        def distance_to_next_holiday(date):
            """Calcule le nombre de jours jusqu'au prochain jour férié."""
            date_only = date.date()  # Convertir pd.Timestamp en datetime.date
            
            # Si c'est déjà un jour férié, distance = 0
            if date_only in fr_holidays:
                return 0
            
            # Chercher les jours fériés futurs et trier par date
            upcoming = sorted([h for h in fr_holidays if h > date_only])
            
            if not upcoming:
                return 90  # limite arbitraire si aucun férié à venir
            
            # Prendre le plus proche et limiter à 90 jours max
            days_until = (upcoming[0] - date_only).days
            return min(days_until, 90)
        
        df["days_to_holiday"] = df["stay_date"].apply(distance_to_next_holiday)
        
        # 4. Vacances scolaires françaises (toutes zones confondues)
        logger.info("Ajout des vacances scolaires...")
        if SCHOOL_HOLIDAYS_AVAILABLE:
            school_holidays = SchoolHolidayDates()
            
            def is_school_holiday(date):
                """Vérifie si la date est pendant les vacances scolaires (toutes zones)."""
                d = date.date() if hasattr(date, 'date') else date
                # Vérifier pour toutes les zones (A, B, C)
                for zone in ['A', 'B', 'C']:
                    if school_holidays.is_holiday_for_zone(d, zone):
                        return 1
                return 0
            
            df["is_vacances_scolaires"] = df["stay_date"].apply(is_school_holiday)
        else:
            logger.warning("⚠️  Librairie vacances-scolaires-france non disponible")
            df["is_vacances_scolaires"] = 0
        
        # 5. Weekends prolongés (3 ou 4 jours)
        logger.info("Ajout des weekends prolongés...")
        
        def is_long_weekend_3days(date):
            """Détecte si la date fait partie d'un weekend de 3 jours."""
            dow = date.dayofweek  # 0=lundi, 6=dimanche
            d = date.date()  # Convertir pd.Timestamp en datetime.date
            
            # Configuration 1: Vendredi-Samedi-Dimanche (férié le vendredi)
            if dow == 4 and d in fr_holidays:  # Vendredi férié
                return 1
            if dow == 5 and (date - pd.Timedelta(days=1)).date() in fr_holidays:  # Samedi après vendredi férié
                return 1
            if dow == 6 and (date - pd.Timedelta(days=2)).date() in fr_holidays:  # Dimanche après vendredi férié
                return 1
            
            # Configuration 2: Samedi-Dimanche-Lundi (férié le lundi)
            if dow == 0 and d in fr_holidays:  # Lundi férié
                return 1
            if dow == 5 and (date + pd.Timedelta(days=2)).date() in fr_holidays:  # Samedi avant lundi férié
                return 1
            if dow == 6 and (date + pd.Timedelta(days=1)).date() in fr_holidays:  # Dimanche avant lundi férié
                return 1
            
            return 0
        
        def is_long_weekend_4days(date):
            """Détecte si la date fait partie d'un weekend de 4 jours."""
            dow = date.dayofweek  # 0=lundi, 6=dimanche
            d = date.date()  # Convertir pd.Timestamp en datetime.date
            
            # Configuration 1: Jeudi-Vendredi-Samedi-Dimanche (fériés jeudi OU vendredi)
            if dow == 3 and d in fr_holidays:  # Jeudi férié
                return 1
            if dow == 4 and (d in fr_holidays or (date - pd.Timedelta(days=1)).date() in fr_holidays):  # Vendredi férié ou après jeudi férié
                return 1
            if dow == 5 and ((date - pd.Timedelta(days=1)).date() in fr_holidays or (date - pd.Timedelta(days=2)).date() in fr_holidays):
                return 1
            if dow == 6 and ((date - pd.Timedelta(days=2)).date() in fr_holidays or (date - pd.Timedelta(days=3)).date() in fr_holidays):
                return 1
            
            # Configuration 2: Samedi-Dimanche-Lundi-Mardi (fériés lundi OU mardi)
            if dow == 1 and d in fr_holidays:  # Mardi férié
                return 1
            if dow == 0 and (d in fr_holidays or (date + pd.Timedelta(days=1)).date() in fr_holidays):  # Lundi férié ou avant mardi férié
                return 1
            if dow == 6 and ((date + pd.Timedelta(days=1)).date() in fr_holidays or (date + pd.Timedelta(days=2)).date() in fr_holidays):
                return 1
            if dow == 5 and ((date + pd.Timedelta(days=2)).date() in fr_holidays or (date + pd.Timedelta(days=3)).date() in fr_holidays):
                return 1
            
            return 0
        
        df["is_long_weekend_3j"] = df["stay_date"].apply(is_long_weekend_3days)
        df["is_long_weekend_4j"] = df["stay_date"].apply(is_long_weekend_4days)
        
        # Afficher les statistiques des features jours fériés
        logger.info(f"Features jours fériés et vacances ajoutées. Shape: {df.shape}")
        logger.info(f"   - Jours fériés: {df['is_holiday_fr'].sum()} ({df['is_holiday_fr'].mean()*100:.2f}%)")
        logger.info(f"   - Jours de pont: {df['is_bridge_day'].sum()} ({df['is_bridge_day'].mean()*100:.2f}%)")
        logger.info(f"   - Distance au férié: min={df['days_to_holiday'].min()}, max={df['days_to_holiday'].max()}, moyenne={df['days_to_holiday'].mean():.1f}")
        logger.info(f"   - Vacances scolaires: {df['is_vacances_scolaires'].sum()} ({df['is_vacances_scolaires'].mean()*100:.2f}%)")
        logger.info(f"   - Weekends 3j: {df['is_long_weekend_3j'].sum()} ({df['is_long_weekend_3j'].mean()*100:.2f}%)")
        logger.info(f"   - Weekends 4j: {df['is_long_weekend_4j'].sum()} ({df['is_long_weekend_4j'].mean()*100:.2f}%)")
        
        return df
    
    def create_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Crée les matrices X (features) et y (target).
        
        Args:
            df: DataFrame préparé
            
        Returns:
            Tuple (X, y, df_filtered) où df_filtered contient toutes les colonnes incluant stay_date et hotCode
        """
        logger.info("Création des features et target...")
        
        horizon = self.config['prediction_horizon']
        
        logger.info("--------------------------------")
        logger.info(f"Horizon: {horizon}")
        logger.info("--------------------------------")
        logger.info(f"df.columns: {df.columns}")
        
        # Recalculer nb_observations en fonction de l'horizon
        # nb_observations doit refléter le nombre de points réellement utilisables
        # Pour horizon=0: compter J-60 à J-1 (pas J-0 qui est la cible)
        # Pour horizon=7: compter J-60 à J-8 (pas J-7 à J-0 qui sont trop proches/cibles)
        if "nb_observations" in df.columns:
            # Pour chaque ligne, compter combien de valeurs TO sont disponibles de J-60 à J-(horizon+1)
            to_cols_for_count = [f"J-{i}" for i in range(60, horizon, -1)]
            to_cols_for_count = [c for c in to_cols_for_count if c in df.columns]
            
            # Compter les valeurs non-NaN pour chaque ligne
            if len(to_cols_for_count) > 0:
                df["nb_observations"] = df[to_cols_for_count].notna().sum(axis=1)
            else:
                df["nb_observations"] = 0
            
            logger.info(f"nb_observations recalculé pour horizon={horizon}")
            logger.info(f"   Colonnes TO comptées: J-60 à J-{horizon+1} ({len(to_cols_for_count)} colonnes)")
            logger.info(f"   Minimum: {df['nb_observations'].min()}")
            logger.info(f"   Maximum: {df['nb_observations'].max()}")
            logger.info(f"   Moyenne: {df['nb_observations'].mean():.1f}")
        
        # Colonnes de TO utilisées comme features : J-60 -> J-(HORIZON+1)
        # Pour horizon=0: utiliser J-60 à J-1 (pas J-0)
        # Pour horizon=7: utiliser J-60 à J-8 (pas J-7 à J-0)
        to_feature_cols = [f"J-{i}" for i in range(60, horizon, -1)]
        to_feature_cols = [c for c in to_feature_cols if c in df.columns]
        
        logger.info(f"Features TO historiques: J-60 à J-{horizon+1} ({len(to_feature_cols)} colonnes)")
        
        if len(to_feature_cols) == 0:
            logger.error(f"❌ Aucune colonne TO disponible comme feature pour horizon={horizon}")
            raise ValueError(f"Pas de colonnes TO disponibles pour horizon={horizon}")
        
        # Features PM compressées
        pm_feature_cols = [
            "pm_mean", "pm_slope", "pm_volatility", "pm_diff_sum",
            "pm_change_ratio", "pm_last_jump", "pm_trend_changes"
        ]
        
        # Features Ant (Anticipation) compressées
        ant_feature_cols = []
        for suffix in ["slope", "volatility", "diff_sum", "change_ratio", "last_jump", "trend_changes"]:
            col = f"ant_{suffix}"
            if col in df.columns:
                ant_feature_cols.append(col)
        
        # Features Ds (Durée de séjour) compressées
        ds_feature_cols = []
        for suffix in ["slope", "volatility", "diff_sum", "change_ratio", "last_jump", "trend_changes"]:
            col = f"ds_{suffix}"
            if col in df.columns:
                ds_feature_cols.append(col)
        
        # Features prix concurrents (médian uniquement)
        comp_feature_cols = []
        for suffix in ["slope", "volatility", "diff_sum", "change_ratio", "last_jump", "trend_changes"]:
            col = f"comp_{suffix}"
            if col in df.columns:
                comp_feature_cols.append(col)
        
        # Features Gap et Elasticity (PM vs Concurrents)
        gap_elasticity_cols = []
        for col in ["gap_last", "gap_slope", "elasticity_last", "elasticity_slope"]:
            if col in df.columns:
                gap_elasticity_cols.append(col)
        
        # Autres features
        other_feature_cols = []
        for col in ["nb_observations", "cluster", "month", "dayofweek", "ToF1",
                    "is_holiday_fr", "is_bridge_day", "days_to_holiday",
                    "is_vacances_scolaires", "is_long_weekend_3j", "is_long_weekend_4j"]:
            if col in df.columns:
                other_feature_cols.append(col)
        
        # Construction de la liste finale de features
        self.feature_cols = to_feature_cols + pm_feature_cols + ant_feature_cols + ds_feature_cols + comp_feature_cols + gap_elasticity_cols + other_feature_cols
        
        logger.info(f"Nombre total de features: {len(self.feature_cols)}")
        logger.info(f"   - TO historiques: {len(to_feature_cols)}")
        logger.info(f"   - Features PM: {len(pm_feature_cols)}")
        logger.info(f"   - Features Ant (anticipation): {len(ant_feature_cols)}")
        logger.info(f"   - Features Ds (durée séjour): {len(ds_feature_cols)}")
        logger.info(f"   - Features concurrents: {len(comp_feature_cols)}")
        logger.info(f"   - Features Gap/Elasticity: {len(gap_elasticity_cols)}")
        logger.info(f"   - Autres features: {len(other_feature_cols)}")
        
        # Cible = TO final J-0
        if "J-0" not in df.columns:
            raise ValueError("La colonne 'J-0' (TO final) est absente du DataFrame")
        
        X = df[self.feature_cols].copy()
        y = df["J-0"].copy()
        
        # Drop des lignes avec NaN
        mask_valid = X.notna().all(axis=1) & y.notna()
        X = X[mask_valid]
        y = y[mask_valid]
        
        # Sauvegarder les données d'entraînement avec stay_date et hotCode
        df_filtered = df[mask_valid].copy()
        self._save_training_data(df_filtered, self.feature_cols)
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Retourner aussi le DataFrame filtré pour la sauvegarde des prédictions de test
        return X, y, df_filtered
    
    def _save_training_data(self, df_complete: pd.DataFrame, feature_cols: List[str], output_dir: str = None):
        """
        Sauvegarde le DataFrame d'entraînement filtré avant normalisation.
        
        Args:
            df_complete: DataFrame complet brut (toutes les colonnes)
            feature_cols: Liste des colonnes features à sauvegarder
            output_dir: Répertoire de sortie
        """
        try:
            # Utiliser le répertoire de sortie de la config si non spécifié
            if output_dir is None:
                output_dir = self._get_output_dir()
            
            # Créer le répertoire s'il n'existe pas
            os.makedirs(output_dir, exist_ok=True)
            
            # Sélectionner les colonnes à sauvegarder
            cols_to_save = []
            
            # Ajouter les colonnes d'identification si elles existent
            if "stay_date" in df_complete.columns:
                cols_to_save.append("stay_date")
            if "hotCode" in df_complete.columns:
                cols_to_save.append("hotCode")
            
            # Ajouter toutes les features
            cols_to_save.extend(feature_cols)
            
            # Ajouter le target
            if "J-0" in df_complete.columns:
                cols_to_save.append("J-0")
            
            # Filtrer le DataFrame pour ne garder que les colonnes pertinentes
            df_to_save = df_complete[cols_to_save].copy()
            
            # Chemin de sauvegarde
            output_path = os.path.join(output_dir, "training_data_before_scaling.csv")
            
            # Sauvegarder le DataFrame filtré en CSV
            df_to_save.to_csv(output_path, index=False, sep=';', encoding='utf-8')
            
            logger.info(f"💾 Données d'entraînement sauvegardées: {output_path}")
            logger.info(f"   Shape: {df_to_save.shape}")
            logger.info(f"   Colonnes: stay_date, hotCode, {len(feature_cols)} features, J-0")
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors de la sauvegarde des données d'entraînement: {e}")
    
    def _save_test_predictions(self, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, df_test_info: pd.DataFrame = None, output_dir: str = None):
        """
        Sauvegarde le DataFrame de test avec les prédictions.
        
        Args:
            X_test: Features de test (non normalisées)
            y_test: Vraies valeurs de test
            y_pred: Prédictions du modèle
            df_test_info: DataFrame avec stay_date et hotCode (optionnel)
            output_dir: Répertoire de sortie
        """
        try:
            # Utiliser le répertoire de sortie de la config si non spécifié
            if output_dir is None:
                output_dir = self._get_output_dir()
            
            # Créer le répertoire s'il n'existe pas
            os.makedirs(output_dir, exist_ok=True)
            
            # Créer le DataFrame de test
            df_test = pd.DataFrame()
            
            # Ajouter stay_date et hotCode en premier si disponibles
            if df_test_info is not None:
                df_test['stay_date'] = df_test_info['stay_date'].values
                df_test['hotCode'] = df_test_info['hotCode'].values
            
            # Ajouter toutes les features
            for col in X_test.columns:
                df_test[col] = X_test[col].values
            
            # Ajouter les prédictions et erreurs
            df_test['y_test'] = y_test.values
            df_test['y_pred'] = y_pred
            df_test['error'] = df_test['y_pred'] - df_test['y_test']
            df_test['abs_error'] = np.abs(df_test['error'])
            
            # Chemin de sauvegarde
            output_path = os.path.join(output_dir, "test_predictions.csv")
            
            # Sauvegarder en CSV
            df_test.to_csv(output_path, index=False, sep=';', encoding='utf-8')
            
            cols_info = "stay_date, hotCode, " if df_test_info is not None else ""
            logger.info(f"💾 Prédictions de test sauvegardées: {output_path}")
            logger.info(f"   Shape: {df_test.shape}")
            logger.info(f"   Colonnes: {cols_info}{len(X_test.columns)} features + y_test + y_pred + error + abs_error")
            logger.info(f"   MAE moyen: {df_test['abs_error'].mean():.4f}")
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors de la sauvegarde des prédictions de test: {e}")
    
    def hyperparameter_search(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Recherche les meilleurs hyperparamètres pour le modèle XGBoost avec RandomizedSearchCV.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionnaire avec les meilleurs paramètres trouvés
        """
        logger.info("=" * 80)
        logger.info("RECHERCHE D'HYPERPARAMÈTRES (RANDOMIZED SEARCH)")
        logger.info("=" * 80)
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/validation
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape}, Validation: {X_val.shape}")
        
        # Grille de paramètres à tester
        param_distributions = {
            'n_estimators': [300, 500, 600, 800],
            'learning_rate': [0.01, 0.03, 0.05, 0.08],
            'max_depth': [5, 6, 7, 8, 9],
            'subsample': [0.8, 0.85, 0.9, 0.95],
            'colsample_bytree': [0.8, 0.85, 0.9, 0.95],
            'min_child_weight': [1, 2, 3],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0]
        }
        
        # Modèle de base
        base_model = xgb.XGBRegressor(
            random_state=random_state,
            n_jobs=-1
        )
        
        # Configuration de la recherche
        n_iter = self.config.get('hyperparam_search', {}).get('n_iter', 30)
        cv_folds = self.config.get('hyperparam_search', {}).get('cv_folds', 3)
        
        logger.info(f"Configuration de la recherche:")
        logger.info(f"   - Nombre d'itérations: {n_iter}")
        logger.info(f"   - Cross-validation: {cv_folds} folds")
        logger.info(f"   - Nombre total d'entraînements: {n_iter * cv_folds}")
        logger.info(f"   - Métrique: neg_mean_absolute_error")
        
        # Recherche randomisée
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        logger.info("\nDémarrage de la recherche d'hyperparamètres...")
        random_search.fit(X_train, y_train)
        
        # Meilleurs paramètres
        best_params = random_search.best_params_
        best_score = -random_search.best_score_  # Convertir neg_MAE en MAE
        
        logger.info("=" * 80)
        logger.info("RÉSULTATS DE LA RECHERCHE")
        logger.info("=" * 80)
        logger.info(f"Meilleur score (MAE sur CV): {best_score:.4f}")
        logger.info(f"Meilleurs paramètres trouvés:")
        for param, value in best_params.items():
            logger.info(f"   - {param}: {value}")
        
        # Évaluation sur le set de validation
        best_model = random_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        logger.info(f"\nPerformances sur validation:")
        logger.info(f"   - MAE: {val_mae:.4f}")
        logger.info(f"   - R²: {val_r2:.4f}")
        logger.info("=" * 80)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'cv_results': random_search.cv_results_
        }
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, df_complete: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X: Features
            y: Target
            df_complete: DataFrame complet avec stay_date et hotCode (optionnel, pour sauvegarde test)
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        logger.info("Entraînement du modèle XGBoost...")
        
        # Split train/test AVANT normalisation (pour garder les noms de colonnes)
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
        
        # Normalisation des features APRÈS le split
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        
        # Configuration du modèle
        model_params = self.config.get('model_params', {
            'n_estimators': 600,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'random_state': random_state
        })
        
        self.model = xgb.XGBRegressor(**model_params)
        
        # Entraînement
        self.model.fit(X_train, y_train)
        
        logger.info("Modèle entraîné avec succès")
        
        # Prédictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Sauvegarder les prédictions de test avec stay_date et hotCode
        if df_complete is not None:
            # Récupérer stay_date et hotCode pour le set de test
            test_indices = X_test_raw.index
            df_test_info = df_complete.loc[test_indices, ['stay_date', 'hotCode']].copy() if 'stay_date' in df_complete.columns else None
            self._save_test_predictions(X_test_raw, y_test, y_pred_test, df_test_info)
        else:
            self._save_test_predictions(X_test_raw, y_test, y_pred_test, None)
        
        # Métriques
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': train_r2
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': test_r2
            },
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'feature_importance': self._get_feature_importance()
        }
        
        self.results = results
        
        # Validation du R²
        if test_r2 < -1 or test_r2 > 1.1:  # Tolérance de 1.1 pour arrondi
            logger.warning(f"⚠️  ATTENTION: R² test anormal ({test_r2:.6f})")
            logger.warning(f"   Statistiques y_test: min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}")
            logger.warning(f"   Statistiques y_pred: min={y_pred_test.min():.4f}, max={y_pred_test.max():.4f}, mean={y_pred_test.mean():.4f}")
        
        logger.info(f"📊 Test MAE: {results['test']['mae']:.4f}")
        logger.info(f"📊 Test R²: {results['test']['r2']:.4f}")
        
        return results
    
    def _get_feature_importance(self) -> pd.DataFrame:
        """
        Récupère l'importance des features.
        
        Returns:
            DataFrame avec les features et leur importance
        """
        importances = self.model.feature_importances_
        feat_importance = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        return feat_importance
    
    def evaluate_model(self, save_plots: bool = True):
        """
        Évalue le modèle et génère des visualisations.
        
        Args:
            save_plots: Si True, sauvegarde les graphiques
        """
        logger.info("Évaluation du modèle...")
        
        if not self.results:
            logger.warning("Aucun résultat disponible. Entraînez d'abord le modèle.")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Scatter plot réel vs prédit
        fig, ax = plt.subplots(figsize=(8, 8))
        y_test = self.results['y_test']
        y_pred = self.results['y_pred_test']
        
        ax.scatter(y_test, y_pred, alpha=0.4)
        ax.plot([0, 1], [0, 1], "--", linewidth=2, color='red')
        ax.set_xlabel("TO réel (J-0)")
        ax.set_ylabel("TO prédit (J-0)")
        ax.set_title("Prédiction du TO final - XGBoost")
        ax.grid(True)
        
        if save_plots:
            results_dir = self._get_output_dir()
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, "xgb_scatter_plot.png")
            plt.savefig(plot_path)
            logger.info(f"📈 Graphique sauvegardé: {plot_path}")
        
        plt.close()
        
        # 2. Feature importance
        feat_imp = self.results['feature_importance'].head(20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_imp["feature"], feat_imp["importance"])
        ax.invert_yaxis()
        ax.set_title("Importance des variables - XGBoost (TOP 20)")
        ax.set_xlabel("Importance")
        
        if save_plots:
            results_dir = self._get_output_dir()
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, "xgb_feature_importance.png")
            plt.savefig(plot_path, bbox_inches='tight')
            logger.info(f"📈 Graphique sauvegardé: {plot_path}")
        
        plt.close()
    
    def save_model_locally(self, model_dir: str = None):
        """
        Sauvegarde le modèle et le scaler localement.
        
        Args:
            model_dir: Répertoire de sauvegarde (optionnel)
        """
        # Utiliser le répertoire de sortie de la config si non spécifié
        if model_dir is None:
            output_dir = self._get_output_dir()
            model_dir = os.path.join(output_dir, "models")
        
        logger.info(f"Sauvegarde locale du modèle dans {model_dir}...")
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "xgb_to_predictor.joblib")
        scaler_path = os.path.join(model_dir, "xgb_scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Sauvegarder aussi la liste des features
        features_path = os.path.join(model_dir, "feature_columns.txt")
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        
        logger.info(f"💾 Modèle sauvegardé: {model_path}")
        logger.info(f"💾 Scaler sauvegardé: {scaler_path}")
        logger.info(f"💾 Features sauvegardées: {features_path}")
        
        return model_path, scaler_path, features_path
    
    def save_to_azure_blob(self, local_paths: List[str], container_name: str = "ml-models"):
        """
        Sauvegarde les fichiers dans Azure Blob Storage avec la structure predictTo/{hotel}/J-{horizon}/.
        
        Args:
            local_paths: Liste des chemins de fichiers locaux à uploader
            container_name: Nom du container Azure
        """
        try:
            # Récupérer la connection string depuis les variables d'environnement
            # (peut provenir d'une variable système ou d'un fichier .env)
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
            if not connection_string:
                logger.warning("⚠️  AZURE_STORAGE_CONNECTION_STRING non définie. Sauvegarde Azure ignorée.")
                return
            
            logger.info(f"☁️  Sauvegarde dans Azure Blob Storage...")
            logger.info(f"   Container: {container_name}")
            
            # Créer le client Blob
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Créer le container s'il n'existe pas
            try:
                container_client = blob_service_client.get_container_client(container_name)
                container_client.get_container_properties()
                logger.info(f"   Container existant trouvé")
            except ResourceNotFoundError:
                container_client = blob_service_client.create_container(container_name)
                logger.info(f"   Nouveau container créé")
            
            # Construire le chemin de base dans Azure : predictTo/{hotel}/J-{horizon}/
            horizon = self.config.get('prediction_horizon', 7)
            hotel_folder = self.hotel_code if self.hotel_code else "ALL"
            azure_base_path = f"predictTo/{hotel_folder}/J-{horizon}"
            
            logger.info(f"   Chemin Azure: {azure_base_path}/")
            
            # Uploader chaque fichier
            uploaded_count = 0
            for local_path in local_paths:
                if not os.path.exists(local_path):
                    logger.warning(f"   ⚠️  Fichier {local_path} non trouvé, ignoré")
                    continue
                
                # Déterminer le chemin relatif du fichier
                # Si le fichier est dans un sous-dossier "models", le préserver
                filename = os.path.basename(local_path)
                parent_dir = os.path.basename(os.path.dirname(local_path))
                
                if parent_dir == "models":
                    blob_name = f"{azure_base_path}/models/{filename}"
                else:
                    blob_name = f"{azure_base_path}/{filename}"
                
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                    uploaded_count += 1
                    logger.info(f"   ✓ {blob_name}")
            
            logger.info(f"✅ {uploaded_count} fichier(s) uploadé(s) dans Azure Blob Storage")
            logger.info(f"   URL: {container_name}/{azure_base_path}/")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde Azure: {e}")
            raise


def load_config(config_path: str = "config_predictTo.yaml") -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML ou utilise les valeurs par défaut.
    
    Args:
        config_path: Chemin du fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    default_config = {
        'clustering_base_dir': '../cluster/results',
        'clustering_results_path': '../cluster/results/{hotCode}/clustering_results.csv',
        'indicateurs_path': '../data/{hotCode}/Indicateurs.csv',
        'rateShopper_path': '../data/{hotCode}/rateShopper.csv',
        'prediction_horizon': 7,
        'test_size': 0.2,
        'random_state': 42,
        'model_params': {
            'n_estimators': 600,
            'learning_rate': 0.05,
            'max_depth': 7,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'random_state': 42
        },
        'azure_container': 'ml-models',
        'save_to_azure': True,
        'output_base_dir': 'results',  # Sera complété par {hotCode} dans _get_output_dir()
        'hyperparam_search': {
            'n_iter': 30,
            'cv_folds': 3
        }
    }
    
    if YAML_AVAILABLE and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Mapper le format YAML vers le format attendu
            config = {
                'clustering_base_dir': yaml_config['data'].get('clustering_base_dir', '../cluster/results'),
                'clustering_results_path': yaml_config['data']['clustering_results'],
                'indicateurs_path': yaml_config['data']['indicateurs'],
                'rateShopper_path': yaml_config['data'].get('rateShopper', '../data/{hotCode}/rateShopper.csv'),
                'prediction_horizon': yaml_config['prediction']['horizon'],
                'test_size': yaml_config['training']['test_size'],
                'random_state': yaml_config['training']['random_state'],
                'model_params': {
                    **yaml_config['model'],
                    'random_state': yaml_config['training']['random_state']
                },
                'azure_container': yaml_config['azure']['container_name'],
                'save_to_azure': yaml_config['azure']['save_to_blob'],
                'output_base_dir': yaml_config['output'].get('base_dir', 'results/D09'),
                'hyperparam_search': yaml_config.get('hyperparam_search', {'n_iter': 30, 'cv_folds': 3})
            }
            logger.info(f"✅ Configuration chargée depuis {config_path}")
            return config
        except Exception as e:
            logger.warning(f"⚠️  Erreur lors du chargement de {config_path}: {e}")
            logger.info("   Utilisation de la configuration par défaut")
    else:
        if not YAML_AVAILABLE:
            logger.info("ℹ️  Module YAML non disponible, utilisation de la config par défaut")
        logger.info("ℹ️  Utilisation de la configuration par défaut")
    
    return default_config


def main():
    """
    Fonction principale pour l'exécution du pipeline d'entraînement.
    """
    os.chdir("C:\\github\\ml-demande\\demande\\predictTo")
    print("Répertoire courant :", os.getcwd())

    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle XGBoost pour la prédiction du TO"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config_predictTo.yaml',
        help='Chemin du fichier de configuration YAML'
    )
    parser.add_argument(
        '--no-azure', 
        action='store_true',
        help='Désactiver la sauvegarde Azure'
    )
    parser.add_argument(
        '--search-hyperparams',
        action='store_true',
        help='Activer la recherche d\'hyperparamètres avant l\'entraînement'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='Horizon de prédiction en jours (ex: 7 pour J+7). Override la valeur du fichier config.'
    )
    parser.add_argument(
        '--hotel',
        type=str,
        default=None,
        help='Code de l\'hôtel pour entraîner un modèle spécifique (ex: D09). Si non spécifié, entraîne sur tous les hôtels.'
    )
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE XGBOOST")
    logger.info("=" * 80)
    
    # Information sur le chargement des variables d'environnement
    if DOTENV_AVAILABLE:
        env_file = Path('.env')
        if env_file.exists():
            logger.info("✅ Fichier .env détecté et chargé")
        else:
            logger.info("ℹ️  Fichier .env non trouvé (utilisation des variables d'environnement système)")
    else:
        logger.info("ℹ️  Module python-dotenv non disponible (utilisation des variables d'environnement système uniquement)")
    
    # Charger la configuration
    config = load_config(args.config)
    
    # Override Azure si demandé
    if args.no_azure:
        config['save_to_azure'] = False
    
    # Override horizon si fourni
    if args.horizon is not None:
        if args.horizon < 0:
            logger.error("❌ L'horizon doit être un entier positif ou zéro")
            sys.exit(1)
        if args.horizon >= 60:
            logger.error("❌ L'horizon maximum est 59 (car les données vont jusqu'à J-60)")
            logger.error("   Pour prédire à J-60, il faudrait des données jusqu'à J-61 minimum")
            sys.exit(1)
        logger.info(f"🔧 Override de l'horizon: {config['prediction_horizon']} → {args.horizon}")
        config['prediction_horizon'] = args.horizon
    
    try:
        # Initialiser le prédicteur
        predictor = XGBoostOccupancyPredictor(config, hotel_code=args.hotel)
        
        # 1. Charger les données
        clusters, indicateurs = predictor.load_data()
        
        # Filtrer par hôtel si spécifié
        if args.hotel:
            logger.info(f"🏨 Filtrage des données pour l'hôtel: {args.hotel}")
            clusters = clusters[clusters['hotCode'] == args.hotel]
            indicateurs = indicateurs[indicateurs['hotCode'] == args.hotel]
            logger.info(f"   Clusters filtrés: {clusters.shape}")
            logger.info(f"   Indicateurs filtrés: {indicateurs.shape}")
            
            if len(clusters) == 0:
                logger.error(f"❌ Aucune donnée trouvée pour l'hôtel {args.hotel}")
                sys.exit(1)
        
        # 2. Préparer les données
        df = predictor.prepare_data(clusters, indicateurs)
        
        # 3. Créer les features et target
        X, y, df_complete = predictor.create_features_target(df)
        
        # 4. Recherche d'hyperparamètres (optionnel)
        if args.search_hyperparams:
            hyperparam_results = predictor.hyperparameter_search(X, y)
            
            # Mettre à jour la config avec les meilleurs paramètres
            logger.info("\n🔧 Mise à jour de la configuration avec les meilleurs paramètres...")
            config['model_params'].update(hyperparam_results['best_params'])
            config['model_params']['random_state'] = config.get('random_state', 42)
            config['model_params']['n_jobs'] = -1
            predictor.config = config
        
        # 5. Entraîner le modèle (avec les meilleurs paramètres si recherche effectuée)
        results = predictor.train_model(X, y, df_complete)
        
        # 6. Évaluer le modèle
        predictor.evaluate_model(save_plots=True)
        
        # 7. Sauvegarder localement (le répertoire sera construit automatiquement selon hotel/horizon)
        model_paths = list(predictor.save_model_locally())
        
        # 8. Sauvegarder dans Azure (si activé)
        if config.get('save_to_azure', False):
            container_name = config.get('azure_container', 'ml-models')
            
            # Collecter tous les fichiers à uploader (modèles + graphiques + CSV)
            output_dir = predictor._get_output_dir()
            files_to_upload = list(model_paths)
            
            # Ajouter les graphiques
            scatter_plot = os.path.join(output_dir, "xgb_scatter_plot.png")
            importance_plot = os.path.join(output_dir, "xgb_feature_importance.png")
            if os.path.exists(scatter_plot):
                files_to_upload.append(scatter_plot)
            if os.path.exists(importance_plot):
                files_to_upload.append(importance_plot)
            
            # Ajouter les CSV
            training_csv = os.path.join(output_dir, "training_data_before_scaling.csv")
            test_csv = os.path.join(output_dir, "test_predictions.csv")
            if os.path.exists(training_csv):
                files_to_upload.append(training_csv)
            if os.path.exists(test_csv):
                files_to_upload.append(test_csv)
            
            logger.info(f"\n📤 Préparation de l'upload Azure ({len(files_to_upload)} fichiers)...")
            predictor.save_to_azure_blob(files_to_upload, container_name)
        
        logger.info("=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)
        
        # Afficher les métriques finales
        logger.info("\n📊 MÉTRIQUES FINALES:")
        logger.info(f"   Train MAE:  {results['train']['mae']:.4f}")
        logger.info(f"   Train RMSE: {results['train']['rmse']:.4f}")
        logger.info(f"   Train R²:   {results['train']['r2']:.4f}")
        logger.info(f"   Test MAE:   {results['test']['mae']:.4f}")
        logger.info(f"   Test RMSE:  {results['test']['rmse']:.4f}")
        logger.info(f"   Test R²:    {results['test']['r2']:.4f}")
        
        # Vérification supplémentaire du R²
        if results['test']['r2'] < 0:
            logger.warning(f"\n⚠️  R² négatif détecté ! Le modèle performe moins bien qu'une simple moyenne.")
        elif results['test']['r2'] > 1:
            logger.error(f"\n❌ R² > 1 détecté ! Problème potentiel dans les données ou le calcul.")
        
    except Exception as e:
        logger.error(f"❌ ERREUR FATALE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

