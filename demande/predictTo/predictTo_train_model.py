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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le prédicteur avec la configuration.
        
        Args:
            config: Dictionnaire de configuration contenant les paramètres du modèle
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.results = {}
        
        logger.info("Initialisation du XGBoostOccupancyPredictor")
        logger.info(f"Configuration: {config}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charge les données de clustering et les indicateurs.
        
        Returns:
            Tuple contenant (clusters_df, indicateurs_df)
        """
        logger.info("Chargement des données...")
        
        try:
            # Charger les résultats de clustering
            clusters = pd.read_csv(
                self.config['clustering_results_path'], 
                sep=';'
            )
            logger.info(f"Clusters chargés: {clusters.shape}")
            
            # Charger les indicateurs
            indicateurs = pd.read_csv(
                self.config['indicateurs_path'], 
                sep=';'
            )
            logger.info(f"Indicateurs chargés: {indicateurs.shape}")
            
            return clusters, indicateurs
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    def compute_pm_features(self, pm_series_raw: pd.Series) -> Dict[str, float]:
        """
        Calcule les features compressées à partir d'une série de PM.
        
        Args:
            pm_series_raw: Série temporelle des prix moyens
            
        Returns:
            Dictionnaire des features calculées
        """
        # Conversion en Series pour utiliser les outils pandas
        s = pd.Series(pm_series_raw)
        
        # Conversion en numérique (float), tout ce qui n'est pas convertible -> NaN
        s = pd.to_numeric(s, errors='coerce')
        
        # Remplacement des +/-inf éventuels
        s = s.replace([np.inf, -np.inf], np.nan)
        
        # Si tout est NaN -> on renvoie des 0 safe
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
        
        # Si après interpolation il reste moins de 2 points valides -> pas de pente possible
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
        
        # Pente (pm_slope)
        x = np.arange(len(arr), dtype=float)
        pm_slope = float(np.polyfit(x, arr, 1)[0])
        
        # Ratio global
        first = arr[0]
        last = arr[-1]
        pm_change_ratio = float((last - first) / first) if first != 0 else 0.0
        
        # Variation récente
        if len(arr) >= 6:
            pm_last_jump = float(last - arr[-6])
        else:
            pm_last_jump = float(last - first)
        
        # Changements de direction
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
        
        # Fusionner avec les clusters
        df = clusters.merge(
            pm_pivot,
            left_on=["hotCode", "stay_date"],
            right_on=["hotCode", "Date"],
            how="left"
        ).drop(columns=["Date"])
        
        logger.info(f"DataFrame fusionné: {df.shape}")
        
        # Calculer les features PM compressées
        pm_cols = [c for c in df.columns if c.startswith("pm_J-")]
        
        # Convertir en numérique
        df[pm_cols] = df[pm_cols].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        
        features_list = []
        for idx, row in df.iterrows():
            pm_series = row[pm_cols].values
            feats = self.compute_pm_features(pm_series)
            features_list.append(feats)
        
        df_feats = pd.DataFrame(features_list)
        df = pd.concat([df, df_feats], axis=1)
        
        logger.info(f"Features PM ajoutées: {df.shape}")
        
        # Ajouter les features temporelles
        if not np.issubdtype(df["stay_date"].dtype, np.datetime64):
            df["stay_date"] = pd.to_datetime(df["stay_date"])
        
        df["month"] = df["stay_date"].dt.month
        df["dayofweek"] = df["stay_date"].dt.dayofweek
        
        return df
    
    def create_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Crée les matrices X (features) et y (target).
        
        Args:
            df: DataFrame préparé
            
        Returns:
            Tuple (X, y)
        """
        logger.info("Création des features et target...")
        
        horizon = self.config['prediction_horizon']
        
        # Colonnes de TO utilisées comme features : J-60 -> J-(HORIZON+1)
        to_feature_cols = [f"J-{i}" for i in range(60, horizon, -1)]
        to_feature_cols = [c for c in to_feature_cols if c in df.columns]
        
        # Features PM compressées
        pm_feature_cols = [
            "pm_mean", "pm_slope", "pm_volatility", "pm_diff_sum",
            "pm_change_ratio", "pm_last_jump", "pm_trend_changes"
        ]
        
        # Autres features
        other_feature_cols = []
        for col in ["nb_observations", "cluster", "month", "dayofweek"]:
            if col in df.columns:
                other_feature_cols.append(col)
        
        # Construction de la liste finale de features
        self.feature_cols = to_feature_cols + pm_feature_cols + other_feature_cols
        
        logger.info(f"Nombre de features: {len(self.feature_cols)}")
        
        # Cible = TO final J-0
        if "J-0" not in df.columns:
            raise ValueError("La colonne 'J-0' (TO final) est absente du DataFrame")
        
        X = df[self.feature_cols].copy()
        y = df["J-0"].copy()
        
        # Drop des lignes avec NaN
        mask_valid = X.notna().all(axis=1) & y.notna()
        X = X[mask_valid]
        y = y[mask_valid]
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        logger.info("Entraînement du modèle XGBoost...")
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split train/test
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
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
        
        # Métriques
        results = {
            'train': {
                'mae': mean_absolute_error(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test': {
                'mae': mean_absolute_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2': r2_score(y_test, y_pred_test)
            },
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'feature_importance': self._get_feature_importance()
        }
        
        self.results = results
        
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
            results_dir = "results"
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
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, "xgb_feature_importance.png")
            plt.savefig(plot_path, bbox_inches='tight')
            logger.info(f"📈 Graphique sauvegardé: {plot_path}")
        
        plt.close()
    
    def save_model_locally(self, model_dir: str = "results/models"):
        """
        Sauvegarde le modèle et le scaler localement.
        
        Args:
            model_dir: Répertoire de sauvegarde
        """
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
    
    def save_to_azure_blob(self, local_paths: List[str], container_name: str = "prediction-demande"):
        """
        Sauvegarde les fichiers dans Azure Blob Storage.
        
        Args:
            local_paths: Liste des chemins de fichiers locaux à uploader
            container_name: Nom du container Azure
        """
        try:
            # Récupérer la connection string depuis les variables d'environnement
            # (peut provenir d'une variable système ou d'un fichier .env)
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            
            if not connection_string:
                logger.warning("AZURE_STORAGE_CONNECTION_STRING non définie. Sauvegarde Azure ignorée.")
                return
            
            # Créer le client Blob
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Créer le container s'il n'existe pas
            try:
                container_client = blob_service_client.get_container_client(container_name)
                container_client.get_container_properties()
            except ResourceNotFoundError:
                container_client = blob_service_client.create_container(container_name)
            
            # Uploader chaque fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for local_path in local_paths:
                if not os.path.exists(local_path):
                    logger.warning(f"Fichier {local_path} non trouvé, ignoré")
                    continue
                
                filename = os.path.basename(local_path)
                blob_name = f"models/{timestamp}/{filename}"
                
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, 
                    blob=blob_name
                )
                
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
            
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
        'clustering_results_path': '../results/clustering_results.csv',
        'indicateurs_path': '../data/Indicateurs.csv',
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
        'azure_container': 'prediction-demande',
        'save_to_azure': True,
        'model_dir': 'results/models'
    }
    
    if YAML_AVAILABLE and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Mapper le format YAML vers le format attendu
            config = {
                'clustering_results_path': yaml_config['data']['clustering_results'],
                'indicateurs_path': yaml_config['data']['indicateurs'],
                'prediction_horizon': yaml_config['prediction']['horizon'],
                'test_size': yaml_config['training']['test_size'],
                'random_state': yaml_config['training']['random_state'],
                'model_params': {
                    **yaml_config['model'],
                    'random_state': yaml_config['training']['random_state']
                },
                'azure_container': yaml_config['azure']['container_name'],
                'save_to_azure': yaml_config['azure']['save_to_blob'],
                'model_dir': yaml_config['output']['model_dir']
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
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle XGBoost pour la prédiction du TO"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config_xgboost.yaml',
        help='Chemin du fichier de configuration YAML'
    )
    parser.add_argument(
        '--no-azure', 
        action='store_true',
        help='Désactiver la sauvegarde Azure'
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
    
    try:
        # Initialiser le prédicteur
        predictor = XGBoostOccupancyPredictor(config)
        
        # 1. Charger les données
        clusters, indicateurs = predictor.load_data()
        
        # 2. Préparer les données
        df = predictor.prepare_data(clusters, indicateurs)
        
        # 3. Créer les features et target
        X, y = predictor.create_features_target(df)
        
        # 4. Entraîner le modèle
        results = predictor.train_model(X, y)
        
        # 5. Évaluer le modèle
        predictor.evaluate_model(save_plots=True)
        
        # 6. Sauvegarder localement
        model_dir = config.get('model_dir', 'results/models')
        local_paths = list(predictor.save_model_locally(model_dir))
        
        # 7. Sauvegarder dans Azure (si activé)
        if config.get('save_to_azure', False):
            container_name = config.get('azure_container', 'prediction-demande')
            predictor.save_to_azure_blob(local_paths, container_name)
        
        logger.info("=" * 80)
        logger.info("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
        logger.info("=" * 80)
        
        # Afficher les métriques finales
        logger.info("\n📊 MÉTRIQUES FINALES:")
        logger.info(f"   Train MAE: {results['train']['mae']:.4f}")
        logger.info(f"   Train R²:  {results['train']['r2']:.4f}")
        logger.info(f"   Test MAE:  {results['test']['mae']:.4f}")
        logger.info(f"   Test R²:   {results['test']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ ERREUR FATALE: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

