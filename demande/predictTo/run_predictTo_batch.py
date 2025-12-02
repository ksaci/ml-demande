"""
Script batch pour entraîner des modèles XGBoost pour différents horizons de prédiction.

Ce script permet d'entraîner automatiquement des modèles pour un hôtel donné
avec plusieurs horizons de prédiction (J-60, J-45, J-30, J-21, J-15, J-10, J-7, J-5, J-3, J-1, J-0).

Usage:
    python run_predictTo_batch.py --hotel D09
    python run_predictTo_batch.py --hotel D09 --horizons 7 14 30
    python run_predictTo_batch.py --hotel D09 --no-azure
    python run_predictTo_batch.py --hotel D09 --search-hyperparams

Auteur: Équipe Data Science
Date: 2025
"""

import os
import sys
import logging
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictTo_batch.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PredictToBatchTrainer:
    """
    Classe pour entraîner des modèles XGBoost en batch pour différents horizons.
    """
    
    # Horizons de prédiction par défaut
    # Note : Maximum J-59 car les données vont jusqu'à J-60 et on ne peut pas utiliser J-60 comme feature
    DEFAULT_HORIZONS = [59, 45, 30, 21, 15, 10, 7, 5, 3, 1, 0]
    
    def __init__(self, hotel_code: str, horizons: List[int] = None, 
                 config_file: str = "config_predictTo.yaml",
                 no_azure: bool = False,
                 search_hyperparams: bool = False):
        """
        Initialise le trainer batch.
        
        Args:
            hotel_code: Code de l'hôtel (ex: D09, 6N8)
            horizons: Liste des horizons à entraîner (par défaut: tous)
            config_file: Fichier de configuration YAML
            no_azure: Si True, désactive l'upload Azure
            search_hyperparams: Si True, active la recherche d'hyperparamètres
        """
        self.hotel_code = hotel_code
        self.horizons = horizons if horizons else self.DEFAULT_HORIZONS
        self.config_file = config_file
        self.no_azure = no_azure
        self.search_hyperparams = search_hyperparams
        self.results = []
        
        logger.info(f"Initialisation du batch training pour l'hôtel: {hotel_code}")
        logger.info(f"Horizons à entraîner: {self.horizons}")
        logger.info(f"Nombre total de modèles: {len(self.horizons)}")
    
    def train_single_horizon(self, horizon: int) -> Dict[str, Any]:
        """
        Entraîne un modèle pour un horizon spécifique.
        
        Args:
            horizon: Horizon de prédiction (en jours)
            
        Returns:
            Dictionnaire avec les résultats de l'entraînement
        """
        logger.info("=" * 80)
        logger.info(f"🚀 ENTRAÎNEMENT POUR HORIZON J-{horizon}")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Construire la commande
        cmd = [
            sys.executable,  # Python interpreter
            "predictTo_train_model.py",
            "--hotel", self.hotel_code,
            "--horizon", str(horizon),
            "--config", self.config_file
        ]
        
        if self.no_azure:
            cmd.append("--no-azure")
        
        if self.search_hyperparams:
            cmd.append("--search-hyperparams")
        
        logger.info(f"Commande: {' '.join(cmd)}")
        
        # Exécuter le script
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                logger.info(f"✅ Entraînement J-{horizon} terminé avec succès")
                logger.info(f"⏱️  Durée: {duration:.2f} secondes")
                
                # Extraire les métriques depuis la sortie
                metrics = self._extract_metrics(result.stdout)
                
                return {
                    'horizon': horizon,
                    'status': 'success',
                    'duration': duration,
                    'metrics': metrics,
                    'error': None
                }
            else:
                logger.error(f"❌ Erreur lors de l'entraînement J-{horizon}")
                logger.error(f"Code retour: {result.returncode}")
                logger.error(f"\n{'='*80}")
                logger.error(f"DÉTAILS DE L'ERREUR (STDERR):")
                logger.error(f"{'='*80}")
                if result.stderr and result.stderr.strip():
                    logger.error(result.stderr)
                else:
                    logger.error("(stderr vide)")
                
                logger.error(f"\n{'='*80}")
                logger.error(f"SORTIE STANDARD (STDOUT - dernières 50 lignes):")
                logger.error(f"{'='*80}")
                if result.stdout and result.stdout.strip():
                    # Afficher les dernières lignes de stdout pour voir où ça a planté
                    stdout_lines = result.stdout.strip().split('\n')
                    last_lines = stdout_lines[-50:] if len(stdout_lines) > 50 else stdout_lines
                    for line in last_lines:
                        logger.error(line)
                else:
                    logger.error("(stdout vide)")
                logger.error(f"{'='*80}\n")
                
                # Combiner stderr et les dernières lignes de stdout pour l'erreur
                error_message = f"Code retour: {result.returncode}\n"
                if result.stderr:
                    error_message += f"STDERR:\n{result.stderr}\n"
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    last_20 = '\n'.join(stdout_lines[-20:]) if len(stdout_lines) > 20 else result.stdout
                    error_message += f"STDOUT (dernières lignes):\n{last_20}"
                
                # Sauvegarder les logs complets dans un fichier pour débogage
                self._save_error_log(horizon, result.returncode, result.stdout, result.stderr)
                
                return {
                    'horizon': horizon,
                    'status': 'error',
                    'duration': duration,
                    'metrics': None,
                    'error': error_message
                }
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"❌ Exception lors de l'entraînement J-{horizon}")
            logger.error(f"Type d'exception: {type(e).__name__}")
            logger.error(f"Message: {str(e)}")
            
            # Capturer le traceback complet
            tb = traceback.format_exc()
            logger.error(f"Traceback complet:\n{tb}")
            
            return {
                'horizon': horizon,
                'status': 'exception',
                'duration': duration,
                'metrics': None,
                'error': f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{tb}"
            }
    
    def _save_error_log(self, horizon: int, returncode: int, stdout: str, stderr: str):
        """
        Sauvegarde les logs d'erreur dans un fichier pour faciliter le débogage.
        
        Args:
            horizon: Horizon de prédiction
            returncode: Code retour du processus
            stdout: Sortie standard complète
            stderr: Sortie d'erreur complète
        """
        try:
            # Créer le dossier de logs d'erreur
            error_log_dir = Path("error_logs")
            error_log_dir.mkdir(exist_ok=True)
            
            # Nom du fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = error_log_dir / f"error_{self.hotel_code}_J-{horizon}_{timestamp}.log"
            
            # Écrire les logs complets
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"ERREUR D'ENTRAÎNEMENT - LOGS COMPLETS\n")
                f.write(f"=" * 80 + "\n")
                f.write(f"Hôtel: {self.hotel_code}\n")
                f.write(f"Horizon: J-{horizon}\n")
                f.write(f"Code retour: {returncode}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 80 + "\n\n")
                
                f.write(f"STDERR:\n")
                f.write(f"-" * 80 + "\n")
                f.write(stderr if stderr else "(vide)\n")
                f.write(f"\n" + "=" * 80 + "\n\n")
                
                f.write(f"STDOUT (COMPLET):\n")
                f.write(f"-" * 80 + "\n")
                f.write(stdout if stdout else "(vide)\n")
                f.write(f"\n" + "=" * 80 + "\n")
            
            logger.info(f"📄 Logs d'erreur sauvegardés dans: {log_filename}")
            
        except Exception as e:
            logger.warning(f"⚠️  Impossible de sauvegarder les logs d'erreur: {e}")
    
    def _extract_metrics(self, output: str) -> Dict[str, float]:
        """
        Extrait les métriques finales de la sortie du script.
        
        Args:
            output: Sortie standard du script
            
        Returns:
            Dictionnaire avec les métriques (MAE, R²)
        """
        metrics = {}
        
        # Chercher les lignes de métriques finales
        for line in output.split('\n'):
            if 'Test MAE:' in line:
                try:
                    # Extraire après "Test MAE:"
                    mae_value = line.split('Test MAE:')[1].strip()
                    metrics['test_mae'] = float(mae_value)
                except Exception as e:
                    logger.debug(f"Erreur extraction MAE: {e} - Ligne: {line}")
            elif 'Test R²:' in line:
                try:
                    # Extraire après "Test R²:" (et non après le premier ":")
                    r2_value = line.split('Test R²:')[1].strip()
                    metrics['test_r2'] = float(r2_value)
                except Exception as e:
                    logger.debug(f"Erreur extraction R²: {e} - Ligne: {line}")
            elif 'Test R2:' in line and 'test_r2' not in metrics:
                try:
                    # Fallback pour R2 sans accent
                    r2_value = line.split('Test R2:')[1].strip()
                    metrics['test_r2'] = float(r2_value)
                except Exception as e:
                    logger.debug(f"Erreur extraction R2: {e} - Ligne: {line}")
        
        return metrics
    
    def train_all_horizons(self):
        """
        Entraîne des modèles pour tous les horizons configurés.
        """
        logger.info("\n" + "=" * 80)
        logger.info("DÉBUT DU BATCH TRAINING")
        logger.info("=" * 80)
        logger.info(f"Hôtel: {self.hotel_code}")
        logger.info(f"Horizons: {self.horizons}")
        logger.info(f"Nombre de modèles à entraîner: {len(self.horizons)}")
        logger.info("=" * 80 + "\n")
        
        batch_start_time = datetime.now()
        
        # Entraîner pour chaque horizon
        for i, horizon in enumerate(self.horizons, 1):
            logger.info(f"\n📍 Progression: {i}/{len(self.horizons)}")
            result = self.train_single_horizon(horizon)
            self.results.append(result)
        
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        
        # Afficher le résumé
        self._print_summary(batch_duration)
    
    def _print_summary(self, total_duration: float):
        """
        Affiche un résumé des entraînements.
        
        Args:
            total_duration: Durée totale du batch (en secondes)
        """
        logger.info("\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DU BATCH TRAINING")
        logger.info("=" * 80)
        
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = sum(1 for r in self.results if r['status'] in ['error', 'exception'])
        
        logger.info(f"Hôtel: {self.hotel_code}")
        logger.info(f"Total de modèles: {len(self.results)}")
        logger.info(f"✅ Succès: {success_count}")
        logger.info(f"❌ Erreurs: {error_count}")
        logger.info(f"⏱️  Durée totale: {total_duration/60:.2f} minutes ({total_duration:.2f} secondes)")
        logger.info("")
        
        # Détails par horizon
        logger.info("Détails par horizon:")
        logger.info("-" * 80)
        logger.info(f"{'Horizon':<10} {'Statut':<12} {'Durée (s)':<12} {'Test MAE':<12} {'Test R²':<12}")
        logger.info("-" * 80)
        
        for result in self.results:
            horizon_str = f"J-{result['horizon']}"
            status = "✅ Succès" if result['status'] == 'success' else "❌ Erreur"
            duration_str = f"{result['duration']:.2f}"
            
            if result['metrics']:
                mae_str = f"{result['metrics'].get('test_mae', 'N/A'):.4f}" if 'test_mae' in result['metrics'] else "N/A"
                r2_str = f"{result['metrics'].get('test_r2', 'N/A'):.4f}" if 'test_r2' in result['metrics'] else "N/A"
            else:
                mae_str = "N/A"
                r2_str = "N/A"
            
            logger.info(f"{horizon_str:<10} {status:<12} {duration_str:<12} {mae_str:<12} {r2_str:<12}")
        
        logger.info("-" * 80)
        
        # Afficher les erreurs s'il y en a
        if error_count > 0:
            logger.info("\n⚠️  ERREURS DÉTAILLÉES:")
            logger.info("=" * 80)
            for result in self.results:
                if result['status'] != 'success':
                    logger.info(f"\n🔴 Horizon J-{result['horizon']} ({result['status']}):")
                    logger.info("-" * 80)
                    if result['error']:
                        # Afficher l'erreur avec une indentation pour la lisibilité
                        error_lines = result['error'].split('\n')
                        for line in error_lines:
                            if line.strip():  # Ignorer les lignes vides
                                logger.info(f"   {line}")
                    else:
                        logger.info("   (Aucun détail d'erreur disponible)")
                    logger.info("-" * 80)
            logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        if error_count == 0:
            logger.info("✅ BATCH TRAINING TERMINÉ AVEC SUCCÈS")
        else:
            logger.info(f"⚠️  BATCH TRAINING TERMINÉ AVEC {error_count} ERREUR(S)")
        logger.info("=" * 80)


def main():
    """
    Fonction principale pour l'exécution du batch training.
    """
    # Changer le répertoire de travail vers le dossier predictTo
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    logger.info(f"Répertoire de travail: {os.getcwd()}")
    
    # Parser les arguments
    parser = argparse.ArgumentParser(
        description="Entraînement batch de modèles XGBoost pour différents horizons"
    )
    parser.add_argument(
        '--hotel',
        type=str,
        required=True,
        help='Code de l\'hôtel (ex: D09, 6N8)'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=None,
        help='Liste des horizons à entraîner (ex: --horizons 7 14 30). Par défaut: tous les horizons standard'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_predictTo.yaml',
        help='Fichier de configuration YAML'
    )
    parser.add_argument(
        '--no-azure',
        action='store_true',
        help='Désactiver la sauvegarde Azure'
    )
    parser.add_argument(
        '--search-hyperparams',
        action='store_true',
        help='Activer la recherche d\'hyperparamètres pour chaque horizon'
    )
    
    args = parser.parse_args()
    
    # Créer et exécuter le trainer
    trainer = PredictToBatchTrainer(
        hotel_code=args.hotel,
        horizons=args.horizons,
        config_file=args.config,
        no_azure=args.no_azure,
        search_hyperparams=args.search_hyperparams
    )
    
    trainer.train_all_horizons()


if __name__ == "__main__":
    main()

