"""
Script de clustering pour analyser les courbes de montée en charge du taux d'occupation d'un hôtel.
Objectif : Identifier les types de montées en charge pour les dates de séjour (J-30 à J).
Taux d'occupation (To) = Chambres vendues (Chv) / (Chambres construites - Te - Gt)
Où : Te = travaux d'entretien, Gt = grands travaux

Usage:
    python prediction_cluster.py <hotCode>
    
Exemple:
    python prediction_cluster.py D09
    
    Charge les données depuis : data/D09/Indicateurs.csv
    Sauvegarde les résultats dans : results/D09/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans  # TimeSeriesKMeans avec DTW pour séries temporelles
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter  # Filtre Savitzky-Golay pour réduire le bruit
from datetime import datetime, timedelta
import os
import warnings
import io
import pickle
warnings.filterwarnings('ignore')

# Import pour Azure Blob Storage
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceNotFoundError, AzureError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Import optionnel pour dotenv (chargement du fichier .env)
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
    # Charger le fichier .env s'il existe
    load_dotenv()
except ImportError:
    DOTENV_AVAILABLE = False

# Configuration du style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def get_optimal_n_jobs(force_parallel=False):
    """
    Détermine le nombre optimal de jobs pour le parallélisme.
    Gère les problèmes de compatibilité Windows/multiprocessing.
    
    Args:
        force_parallel (bool): Forcer n_jobs=-1 même sur Windows (peut causer des erreurs)
    
    Returns:
        int: Nombre de jobs à utiliser (-1 pour tous les CPU, ou 1 si problème)
    """
    try:
        import platform
        if force_parallel:
            # Forcer le parallélisme si demandé
            return -1
        
        # Sur Windows, le multiprocessing peut avoir des problèmes avec tslearn
        # Par défaut, utiliser 1 CPU pour éviter les erreurs de pickling
        if platform.system() == 'Windows':
            return 1  # Mode sécurisé sur Windows
        else:
            # Sur Linux/Mac, utiliser tous les CPU
            return -1
    except:
        return 1


class HotelBookingClustering:
    """
    Classe pour analyser et clusteriser les courbes de montée en charge des réservations hôtelières.
    """
    
    def __init__(self, hotCode=None, csv_path=None, days_before=30):
        """
        Initialise l'analyse de clustering.
        
        Args:
            hotCode (str, optional): Code de l'hôtel (3 caractères, ex: 'D09'). 
                                      Si spécifié, charge depuis data/{hotCode}/Indicateurs.csv
            csv_path (str, optional): Chemin vers le fichier CSV (priorité si hotCode non spécifié)
            days_before (int): Nombre de jours avant la date de séjour à analyser (par défaut 30)
        """
        self.hotCode = hotCode
        
        # Déterminer le chemin CSV
        if hotCode is not None:
            self.csv_path = f'../data/{hotCode}/Indicateurs.csv'
            self.results_dir = f'results/{hotCode}'
        elif csv_path is not None:
            self.csv_path = csv_path
            self.results_dir = 'results'
        else:
            raise ValueError("Vous devez spécifier soit hotCode soit csv_path")
        
        self.days_before = days_before
        self.df = None
        self.curves_df = None
        self.curves_df_original = None  # Sauvegarde des courbes avant lissage
        self.scaled_curves = None
        self.scaler = TimeSeriesScalerMeanVariance()
        self.ts_kmeans_model = None
        self.optimal_k = None
        
        # Créer le dossier de résultats s'il n'existe pas
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration Azure
        self.azure_enabled = False
        self.blob_service_client = None
        self.azure_container = "ml-models"
        self._init_azure_storage()
    
    def _init_azure_storage(self):
        """
        Initialise la connexion à Azure Blob Storage.
        Lit la connection string depuis les variables d'environnement ou le fichier .env.
        """
        if not AZURE_AVAILABLE:
            print("  ℹ️  SDK Azure non disponible - sauvegarde locale uniquement")
            return
        
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        if not connection_string:
            print("  ℹ️  AZURE_STORAGE_CONNECTION_STRING non définie - sauvegarde locale uniquement")
            return
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Vérifier que le container existe, sinon le créer
            try:
                container_client = self.blob_service_client.get_container_client(self.azure_container)
                container_client.get_container_properties()
            except ResourceNotFoundError:
                self.blob_service_client.create_container(self.azure_container)
                print(f"  ✅ Container Azure '{self.azure_container}' créé")
            
            self.azure_enabled = True
            print(f"  ✅ Connexion Azure Blob Storage établie (container: {self.azure_container})")
            
        except Exception as e:
            print(f"  ⚠️  Erreur lors de la connexion Azure: {e}")
            print("  → Fallback sur sauvegarde locale")
            self.azure_enabled = False
    
    def _save_to_azure(self, file_content, blob_name, content_type='application/octet-stream'):
        """
        Sauvegarde un fichier dans Azure Blob Storage.
        
        Args:
            file_content: Contenu du fichier (bytes ou file-like object)
            blob_name: Nom du blob (chemin dans le container)
            content_type: Type MIME du contenu
            
        Returns:
            bool: True si succès, False sinon
        """
        if not self.azure_enabled:
            return False
        
        try:
            blob_path = f"clustering/{self.hotCode}/{blob_name}"
            blob_client = self.blob_service_client.get_blob_client(
                container=self.azure_container,
                blob=blob_path
            )
            
            # Créer l'objet ContentSettings
            content_settings = ContentSettings(content_type=content_type)
            
            # Upload avec overwrite
            if isinstance(file_content, bytes):
                blob_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)
            else:
                blob_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)
            
            return True
            
        except Exception as e:
            print(f"    ⚠️  Erreur Azure upload ({blob_name}): {e}")
            return False
    
    def _save_file(self, file_path, blob_name, description="Fichier"):
        """
        Sauvegarde un fichier en priorité sur Azure, sinon sur disque local.
        
        Args:
            file_path: Chemin local du fichier
            blob_name: Nom du blob dans Azure
            description: Description du fichier pour les logs
            
        Returns:
            str: Emplacement de sauvegarde ("azure" ou "local")
        """
        # Essayer d'abord Azure
        if self.azure_enabled and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                content = f.read()
            
            if self._save_to_azure(content, blob_name):
                print(f"  ☁️  {description} sauvegardé dans Azure: ml-models/clustering/{self.hotCode}/{blob_name}")
                return "azure"
        
        # Fallback sur disque local (déjà sauvegardé)
        print(f"  💾 {description} sauvegardé localement: {file_path}")
        return "local"
        
    def load_data(self, year_filter=None):
        """
        Charge les données depuis le fichier CSV et calcule le taux d'occupation.
        
        Args:
            year_filter (int, optional): Année à filtrer. Si None, aucune année n'est filtrée (toutes les années).
        """
        print("📊 Chargement des données...")
        if self.hotCode:
            print(f"  📂 Fichier : {self.csv_path} (Hôtel {self.hotCode})")
        else:
            print(f"  📂 Fichier : {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path, sep=';', parse_dates=['Date', 'ObsDate'])
        
        # Filtrer par année si spécifié (uniquement sur Date = date de séjour)
        if year_filter is not None:
            print(f"  🔍 Filtrage sur les dates de séjour de l'année {year_filter}...")
            nb_before = len(self.df)
            
            # Vérifier d'abord si l'année existe
            available_years = sorted(self.df['Date'].dt.year.unique())
            
            if year_filter not in available_years:
                print(f"    ⚠️ ATTENTION : Aucune donnée pour l'année {year_filter} !")
                print(f"    Années disponibles : {available_years}")
                print(f"    → Utilisation de toutes les années disponibles")
            else:
                # Identifier les combinaisons (hotCode, Date) pour les séjours de l'année filtrée
                # On filtre uniquement sur Date (date de séjour), pas sur ObsDate
                mask_stay_year = self.df['Date'].dt.year == year_filter
                stay_combinations = self.df[mask_stay_year][['hotCode', 'Date']].drop_duplicates()
                
                print(f"    - {len(stay_combinations):,} dates de séjour en {year_filter}")
                
                # Garder TOUTES les observations pour ces séjours, même si ObsDate est avant 2024
                # (nécessaire pour les courbes de montée en charge J-60)
                # Exemple : séjour du 1er janvier 2024 → observations depuis début novembre 2023
                self.df = self.df.merge(
                    stay_combinations,
                    on=['hotCode', 'Date'],
                    how='inner'
                )
                
                nb_after = len(self.df)
                print(f"    - {nb_before:,} lignes → {nb_after:,} lignes ({nb_after/nb_before*100:.1f}% conservées)")
                
                # Afficher la période d'observation couverte
                obs_years = sorted(self.df['ObsDate'].dt.year.unique())
                obs_date_min = self.df['ObsDate'].min()
                obs_date_max = self.df['ObsDate'].max()
                print(f"    - Période d'observation : {obs_date_min.strftime('%Y-%m-%d')} à {obs_date_max.strftime('%Y-%m-%d')}")
                if min(obs_years) < year_filter:
                    print(f"      ⚠️ Les observations de {min(obs_years)} sont incluses (nécessaires pour J-{self.days_before})")
        else:
            print(f"  ℹ️ Aucun filtre d'année appliqué - utilisation de toutes les années disponibles")
        
        # Calculer le taux d'occupation (To = Chv / (Construites - Te - Gt))
        # Où : Te = travaux d'entretien, Gt = grands travaux
        print("  - Calcul du taux d'occupation (To = Chv / (Construites - Te - Gt))...")
        print("    - Te = travaux d'entretien, Gt = grands travaux")
        denominator = self.df['Construites'] - self.df['Te'] - self.df['Gt']
        self.df['To'] = self.df['Chv'] / denominator.replace(0, np.nan)
        
        # Remplacer les valeurs infinies par NaN
        self.df['To'] = self.df['To'].replace([np.inf, -np.inf], np.nan)
        
        # Borner le taux d'occupation entre 0 et 1 (ou plus si surbooking)
        print(f"    - Taux d'occupation min/max : {self.df['To'].min():.3f} / {self.df['To'].max():.3f}")
        
        print(f"✓ Données chargées : {len(self.df):,} lignes, {len(self.df.columns)} colonnes")
        print(f"  - Période des dates de séjour : {self.df['Date'].min()} à {self.df['Date'].max()}")
        print(f"  - Période d'observation : {self.df['ObsDate'].min()} à {self.df['ObsDate'].max()}")
        print(f"  - Nombre d'hôtels : {self.df['hotCode'].nunique()}")
        return self
    
    def prepare_booking_curves(self, min_observations=20):
        """
        Prépare les courbes de montée en charge pour chaque date de séjour.
        
        Args:
            min_observations (int): Nombre minimum d'observations requises pour inclure une courbe
            
        Returns:
            pd.DataFrame: DataFrame avec les courbes de montée en charge
        """
        print(f"\n🔄 Préparation des courbes de montée en charge (J-{self.days_before} à J)...")
        
        curves_data = []
        
        # Grouper par hôtel et date de séjour
        grouped = self.df.groupby(['hotCode', 'Date'])
        total_groups = len(grouped)
        
        print(f"  - Nombre total de combinaisons (hôtel, date de séjour) : {total_groups}")
        
        for idx, ((hotel_code, stay_date), group) in enumerate(grouped):
            if (idx + 1) % 1000 == 0:
                print(f"    Traitement : {idx + 1}/{total_groups} courbes...")
            
            # Trier par date d'observation
            group = group.sort_values('ObsDate')
            
            # Calculer le nombre de jours avant la date de séjour
            group['days_before_stay'] = (stay_date - group['ObsDate']).dt.days
            
            # Filtrer pour ne garder que les observations entre J-days_before et J
            group_filtered = group[(group['days_before_stay'] >= 0) & 
                                   (group['days_before_stay'] <= self.days_before)]
            
            # Vérifier qu'on a suffisamment d'observations
            if len(group_filtered) < min_observations:
                continue
            
            # Créer un dictionnaire pour stocker la courbe
            curve_dict = {
                'hotCode': hotel_code,
                'stay_date': stay_date,
                'nb_observations': len(group_filtered)
            }
            
            # Créer une série temporelle de J-30 à J
            for day in range(self.days_before, -1, -1):
                day_data = group_filtered[group_filtered['days_before_stay'] == day]
                
                if len(day_data) > 0:
                    # Prendre la dernière observation de ce jour (taux d'occupation)
                    curve_dict[f'J-{day}'] = day_data.iloc[-1]['To']
                else:
                    # Si pas de données, interpoler ou mettre NaN
                    curve_dict[f'J-{day}'] = np.nan
            
            curves_data.append(curve_dict)
        
        self.curves_df = pd.DataFrame(curves_data)
        
        # Interpoler les valeurs manquantes (interpolation linéaire)
        print("\n🔧 Traitement des valeurs manquantes...")
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        self.curves_df[feature_cols] = self.curves_df[feature_cols].interpolate(
            method='linear', axis=1, limit_direction='both'
        )
        
        # Supprimer les lignes avec encore des NaN
        before_dropna = len(self.curves_df)
        self.curves_df = self.curves_df.dropna(subset=feature_cols)
        after_dropna = len(self.curves_df)
        
        if before_dropna > after_dropna:
            print(f"  - {before_dropna - after_dropna} courbes supprimées (valeurs manquantes)")
        
        print(f"✓ {len(self.curves_df)} courbes complètes préparées")
        
        # Afficher quelques statistiques
        print(f"\n📈 Statistiques du taux d'occupation :")
        to_final = self.curves_df['J-0']
        print(f"  - Moyenne à J : {to_final.mean():.3f} ({to_final.mean()*100:.1f}%)")
        print(f"  - Médiane à J : {to_final.median():.3f} ({to_final.median()*100:.1f}%)")
        print(f"  - Min/Max à J : {to_final.min():.3f} / {to_final.max():.3f}")
        
        return self.curves_df
    
    def apply_smoothing(self, enable=True, window_length=15, polyorder=3, double_pass=False):
        """
        Applique un filtre Savitzky-Golay pour réduire le bruit sur les courbes.
        
        Args:
            enable (bool): Activer ou désactiver le lissage (défaut: True)
            window_length (int): Longueur de la fenêtre du filtre (doit être impair, défaut: 15)
            polyorder (int): Ordre du polynôme (doit être < window_length, défaut: 3)
            double_pass (bool): Appliquer le lissage deux fois pour plus d'efficacité (défaut: False)
        
        Returns:
            pd.DataFrame: DataFrame avec les courbes lissées
        """
        if not enable:
            print("\n🔧 Lissage désactivé - courbes brutes conservées")
            return self.curves_df
        
        # Sauvegarder les courbes originales avant le lissage
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        self.curves_df_original = self.curves_df[['hotCode', 'stay_date', 'nb_observations'] + feature_cols].copy()
        
        print(f"\n🔧 Application du filtre Savitzky-Golay pour réduire le bruit...")
        print(f"  - Fenêtre : {window_length} points")
        print(f"  - Ordre du polynôme : {polyorder}")
        if double_pass:
            print(f"  - Mode : Double passage (lissage renforcé)")
        
        # Vérifier que window_length est impair
        if window_length % 2 == 0:
            window_length += 1
            print(f"  ⚠️ window_length ajusté à {window_length} (doit être impair)")
        
        # Vérifier que window_length < nombre de points
        n_points = len(feature_cols)
        
        if window_length >= n_points:
            window_length = n_points - 1 if n_points % 2 == 0 else n_points - 2
            print(f"  ⚠️ window_length ajusté à {window_length} (trop grand pour {n_points} points)")
        
        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"  ⚠️ polyorder ajusté à {polyorder} (doit être < window_length)")
        
        # Appliquer le filtre sur chaque courbe
        smoothed_curves = []
        curves_with_issues = 0
        
        for idx, row in self.curves_df.iterrows():
            curve_values = row[feature_cols].values
            
            try:
                # Appliquer le filtre Savitzky-Golay
                smoothed_curve = savgol_filter(curve_values, window_length, polyorder)
                
                # Si double_pass, appliquer une deuxième fois pour plus de lissage
                if double_pass:
                    smoothed_curve = savgol_filter(smoothed_curve, window_length, polyorder)
                
                smoothed_curves.append(smoothed_curve)
            except Exception as e:
                # Si erreur (courbe trop courte, valeurs NaN, etc.), garder l'originale
                smoothed_curves.append(curve_values)
                curves_with_issues += 1
        
        # Remplacer les valeurs dans le DataFrame
        smoothed_array = np.array(smoothed_curves)
        for i, col in enumerate(feature_cols):
            self.curves_df[col] = smoothed_array[:, i]
        
        if curves_with_issues > 0:
            print(f"  ⚠️ {curves_with_issues} courbes n'ont pas pu être lissées (conservées telles quelles)")
        
        print(f"✓ Lissage appliqué sur {len(self.curves_df)} courbes")
        
        # Afficher un exemple de comparaison
        if len(self.curves_df) > 0 and self.curves_df_original is not None:
            sample_idx = 0
            original_curve = self.curves_df_original.iloc[sample_idx][feature_cols].values
            smoothed_curve = self.curves_df.iloc[sample_idx][feature_cols].values
            
            # Calculer la différence moyenne
            diff = np.abs(original_curve - smoothed_curve).mean()
            print(f"  📊 Exemple (courbe #{sample_idx}) :")
            print(f"     - Différence moyenne : {diff:.4f}")
            print(f"     - Écart-type original : {original_curve.std():.4f}")
            print(f"     - Écart-type lissé : {smoothed_curve.std():.4f}")
        
        return self.curves_df
    
    def analyze_initial_occupancy(self):
        """
        Analyse les taux d'occupation initiaux (à J-30) pour identifier les cas particuliers.
        """
        print(f"\n🔍 Analyse des taux d'occupation à J-{self.days_before}...\n")
        
        # Récupérer les To à J-30 et J
        to_initial = self.curves_df[f'J-{self.days_before}']
        to_final = self.curves_df['J-0']
        
        print("=" * 70)
        print(f"📊 STATISTIQUES DU TAUX D'OCCUPATION À J-{self.days_before}")
        print("=" * 70)
        print(f"  - Moyenne        : {to_initial.mean():.3f} ({to_initial.mean()*100:.1f}%)")
        print(f"  - Médiane        : {to_initial.median():.3f} ({to_initial.median()*100:.1f}%)")
        print(f"  - Écart-type     : {to_initial.std():.3f}")
        print(f"  - Min / Max      : {to_initial.min():.3f} / {to_initial.max():.3f}")
        print(f"  - Percentile 75% : {to_initial.quantile(0.75):.3f} ({to_initial.quantile(0.75)*100:.1f}%)")
        print(f"  - Percentile 90% : {to_initial.quantile(0.90):.3f} ({to_initial.quantile(0.90)*100:.1f}%)")
        print(f"  - Percentile 95% : {to_initial.quantile(0.95):.3f} ({to_initial.quantile(0.95)*100:.1f}%)")
        
        # Identifier les cas avec To élevé dès J-30
        high_initial_threshold = 0.5  # 50%
        very_high_initial_threshold = 0.7  # 70%
        
        high_initial = self.curves_df[to_initial >= high_initial_threshold]
        very_high_initial = self.curves_df[to_initial >= very_high_initial_threshold]
        
        print(f"\n🔍 DATES DE SÉJOUR AVEC TO ÉLEVÉ DÈS J-{self.days_before} :")
        print(f"  - To ≥ 50% : {len(high_initial)} dates ({len(high_initial)/len(self.curves_df)*100:.1f}%)")
        print(f"  - To ≥ 70% : {len(very_high_initial)} dates ({len(very_high_initial)/len(self.curves_df)*100:.1f}%)")
        
        if len(very_high_initial) > 0:
            print(f"\n⚠️  ATTENTION : {len(very_high_initial)} dates de séjour ont un To ≥ 70% dès J-{self.days_before} !")
            print("   Cela peut indiquer :")
            print("   - Des événements spéciaux (salons, concerts, etc.)")
            print("   - Des périodes de très forte demande (vacances, ponts, etc.)")
            print("   - Des réservations de groupes anticipées")
            
            # Afficher quelques exemples
            print(f"\n📋 Exemples de dates avec To très élevé à J-{self.days_before} :")
            examples = very_high_initial.nlargest(10, f'J-{self.days_before}')[
                ['hotCode', 'stay_date', f'J-{self.days_before}', 'J-0']
            ]
            examples_display = examples.copy()
            examples_display[f'J-{self.days_before}'] = examples_display[f'J-{self.days_before}'].apply(
                lambda x: f"{x:.3f} ({x*100:.1f}%)"
            )
            examples_display['J-0'] = examples_display['J-0'].apply(
                lambda x: f"{x:.3f} ({x*100:.1f}%)"
            )
            examples_display.columns = ['Hôtel', 'Date séjour', f'To à J-{self.days_before}', 'To à J']
            print(examples_display.to_string(index=False))
        
        # Analyser la distribution des To initiaux
        print(f"\n📊 RÉPARTITION DES TO À J-{self.days_before} :")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float('inf')]
        labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                  '50-60%', '60-70%', '70-80%', '80-90%', '90-100%', '>100%']
        
        distribution = pd.cut(to_initial, bins=bins, labels=labels).value_counts().sort_index()
        for category, count in distribution.items():
            percentage = (count / len(to_initial)) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {category:>8} : {count:>6} dates ({percentage:>5.1f}%) {bar}")
        
        # Analyser la croissance (J-30 à J)
        print(f"\n📈 CROISSANCE DU TAUX D'OCCUPATION (J-{self.days_before} → J) :")
        growth = to_final - to_initial
        print(f"  - Croissance moyenne : {growth.mean():.3f} ({growth.mean()*100:.1f} points de %)")
        print(f"  - Croissance médiane : {growth.median():.3f} ({growth.median()*100:.1f} points de %)")
        print(f"  - Croissance min/max : {growth.min():.3f} / {growth.max():.3f}")
        
        # Identifier les cas avec faible croissance (déjà pleins)
        low_growth = self.curves_df[growth < 0.1]  # Croissance < 10 points de %
        print(f"\n  - Dates avec faible croissance (< 10 pts) : {len(low_growth)} ({len(low_growth)/len(self.curves_df)*100:.1f}%)")
        
        if len(low_growth) > 0:
            avg_initial_low_growth = low_growth[f'J-{self.days_before}'].mean()
            print(f"    → To moyen à J-{self.days_before} pour ces dates : {avg_initial_low_growth:.3f} ({avg_initial_low_growth*100:.1f}%)")
        
        # Créer une visualisation
        self._plot_initial_occupancy_analysis(to_initial, to_final, growth)
        
        print("\n" + "=" * 70)
        return self
    
    def _plot_initial_occupancy_analysis(self, to_initial, to_final, growth):
        """
        Crée des visualisations pour l'analyse des taux d'occupation initiaux.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribution des To à J-30
        axes[0, 0].hist(to_initial, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(to_initial.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Moyenne: {to_initial.mean():.3f}')
        axes[0, 0].axvline(to_initial.median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Médiane: {to_initial.median():.3f}')
        axes[0, 0].set_xlabel(f'Taux d\'occupation à J-{self.days_before}', fontsize=9)
        axes[0, 0].set_ylabel('Nombre de dates de séjour', fontsize=9)
        axes[0, 0].set_title(f'Distribution des To à J-{self.days_before}', 
                            fontsize=10, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution des To à J
        axes[0, 1].hist(to_final, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].axvline(to_final.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Moyenne: {to_final.mean():.3f}')
        axes[0, 1].axvline(to_final.median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Médiane: {to_final.median():.3f}')
        axes[0, 1].set_xlabel('Taux d\'occupation à J', fontsize=9)
        axes[0, 1].set_ylabel('Nombre de dates de séjour', fontsize=9)
        axes[0, 1].set_title('Distribution des To à J', fontsize=10, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot : To initial vs To final
        axes[1, 0].scatter(to_initial, to_final, alpha=0.3, s=20, color='purple')
        axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='To initial = To final')
        axes[1, 0].set_xlabel(f'To à J-{self.days_before}', fontsize=9)
        axes[1, 0].set_ylabel('To à J', fontsize=9)
        axes[1, 0].set_title(f'Relation entre To initial (J-{self.days_before}) et To final (J)', 
                            fontsize=10, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, max(to_initial.max(), to_final.max()) + 0.1)
        axes[1, 0].set_ylim(0, max(to_initial.max(), to_final.max()) + 0.1)
        
        # 4. Distribution de la croissance
        axes[1, 1].hist(growth, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[1, 1].axvline(growth.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Moyenne: {growth.mean():.3f}')
        axes[1, 1].axvline(growth.median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Médiane: {growth.median():.3f}')
        axes[1, 1].set_xlabel(f'Croissance du To (J-{self.days_before} → J)', fontsize=9)
        axes[1, 1].set_ylabel('Nombre de dates de séjour', fontsize=9)
        axes[1, 1].set_title('Distribution de la croissance du taux d\'occupation', 
                            fontsize=10, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.results_dir}/initial_occupancy_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'initial_occupancy_analysis.png', 'Graphique analyse initiale')
        
        plt.show()
    
    def normalize_curves(self):
        """
        Normalise les courbes de montée en charge avec TimeSeriesScalerMeanVariance.
        """
        print("\n🔧 Normalisation des courbes avec TimeSeriesScalerMeanVariance...")
        
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        
        # Convertir au format tslearn : (n_samples, n_timestamps, n_features)
        # Pour nous : (n_courbes, n_jours, 1)
        data_3d = self.curves_df[feature_cols].values[:, :, np.newaxis]
        
        # Normaliser
        self.scaled_curves = self.scaler.fit_transform(data_3d)
        
        print(f"✓ Normalisation effectuée : {self.scaled_curves.shape}")
        print(f"  Format tslearn : (n_samples={self.scaled_curves.shape[0]}, n_timestamps={self.scaled_curves.shape[1]}, n_features={self.scaled_curves.shape[2]})")
        return self.scaled_curves
    
    def find_optimal_clusters(self, max_k=15, metric="dtw", sample_size=None):
        """
        Utilise la méthode du coude et le score de silhouette pour trouver le nombre optimal de clusters.
        
        Args:
            max_k (int): Nombre maximum de clusters à tester
            metric (str): Métrique de distance ('dtw', 'euclidean', 'softdtw')
            sample_size (int): Nombre d'échantillons à utiliser (None = tous). Recommandé: 2000-3000 pour DTW
        """
        print(f"\n🔍 Recherche du nombre optimal de clusters avec TimeSeriesKMeans...")
        print(f"  📏 Métrique utilisée : {metric.upper()}")
        
        # Détecter le nombre de CPU disponibles et n_jobs optimal
        n_cpus = os.cpu_count() or 1
        n_jobs = get_optimal_n_jobs()
        
        if n_jobs == -1:
            print(f"  ⚡ Parallélisme activé : n_jobs=-1 (utilise {n_cpus} CPU)")
        else:
            print(f"  ⚙️ Parallélisme : n_jobs={n_jobs} (Windows - mode sécurisé)")
        
        # Si DTW et beaucoup de données, utiliser un échantillon
        data_to_use = self.scaled_curves
        original_size = len(self.scaled_curves)
        
        if sample_size is not None and sample_size < original_size:
            print(f"  ⚡ Utilisation d'un échantillon de {sample_size} courbes (sur {original_size}) pour accélérer")
            indices = np.random.choice(original_size, sample_size, replace=False)
            data_to_use = self.scaled_curves[indices]
        elif metric == "dtw" and original_size > 3000:
            # Réduction automatique pour DTW si trop de données
            sample_size = 2000
            print(f"  ⚡ DTW détecté avec {original_size} courbes → échantillon de {sample_size} pour accélérer")
            indices = np.random.choice(original_size, sample_size, replace=False)
            data_to_use = self.scaled_curves[indices]
        else:
            print(f"  📊 Utilisation de toutes les {original_size} courbes")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        k_range = range(2, max_k + 1)
        
        # Réduire n_init pour DTW (plus lent)
        n_init = 3 if metric == "dtw" else 5
        
        for k in k_range:
            print(f"  🔄 Test K={k}...", end=" ", flush=True)
            
            # Essayer avec n_jobs, fallback sur 1 si erreur
            try:
                ts_kmeans = TimeSeriesKMeans(
                    n_clusters=k, 
                    metric=metric, 
                    random_state=42, 
                    n_init=n_init, 
                    verbose=False,
                    max_iter=50,  # Réduire le nombre d'itérations max
                    n_jobs=n_jobs
                )
                labels = ts_kmeans.fit_predict(data_to_use)
            except Exception as e:
                # Si erreur avec parallélisme, fallback sur n_jobs=1
                if n_jobs != 1:
                    print(f"\n    ⚠️ Erreur avec n_jobs={n_jobs}, fallback sur n_jobs=1...", end=" ", flush=True)
                    ts_kmeans = TimeSeriesKMeans(
                        n_clusters=k, 
                        metric=metric, 
                        random_state=42, 
                        n_init=n_init, 
                        verbose=False,
                        max_iter=50,
                        n_jobs=1
                    )
                    labels = ts_kmeans.fit_predict(data_to_use)
                else:
                    raise e
            
            inertias.append(ts_kmeans.inertia_)
            
            # Pour silhouette et davies-bouldin, on utilise les données en 2D
            scaled_2d = data_to_use.reshape(data_to_use.shape[0], -1)
            silhouette_scores.append(silhouette_score(scaled_2d, labels))
            davies_bouldin_scores.append(davies_bouldin_score(scaled_2d, labels))
            
            print(f"Inertie={ts_kmeans.inertia_:.2f}, "
                  f"Silhouette={silhouette_scores[-1]:.3f}, "
                  f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}")
        
        # Trouver le meilleur k basé sur le score de silhouette
        best_idx = np.argmax(silhouette_scores)
        self.optimal_k = list(k_range)[best_idx]
        
        print(f"\n✓ Nombre optimal de clusters suggéré : {self.optimal_k}")
        print(f"  (Score de silhouette = {silhouette_scores[best_idx]:.3f})")
        
        # Créer les graphiques
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Méthode du coude
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Nombre de clusters (K)', fontsize=9)
        axes[0].set_ylabel('Inertie', fontsize=9)
        axes[0].set_title('Méthode du coude', fontsize=10, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Score de silhouette
        axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(x=self.optimal_k, color='r', linestyle='--', 
                       label=f'Optimal K={self.optimal_k}')
        axes[1].set_xlabel('Nombre de clusters (K)', fontsize=9)
        axes[1].set_ylabel('Score de Silhouette', fontsize=9)
        axes[1].set_title('Score de Silhouette', fontsize=10, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin Score (plus bas = meilleur)
        axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
        axes[2].set_xlabel('Nombre de clusters (K)', fontsize=9)
        axes[2].set_ylabel('Davies-Bouldin Score', fontsize=9)
        axes[2].set_title('Davies-Bouldin Score (plus bas = meilleur)', 
                         fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.results_dir}/clustering_optimal_k.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'clustering_optimal_k.png', 'Graphique K optimal')
        
        plt.show()
        
        return self.optimal_k
    
    def perform_clustering(self, n_clusters=None, metric="dtw", n_init=10):
        """
        Effectue le clustering avec TimeSeriesKMeans.
        
        Args:
            n_clusters (int): Nombre de clusters (utilise optimal_k si None)
            metric (str): Métrique de distance ('dtw', 'euclidean', 'softdtw')
            n_init (int): Nombre d'initialisations (réduire si DTW est lent)
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k else 5
        
        print(f"\n🎯 Clustering avec TimeSeriesKMeans (K={n_clusters}, metric={metric.upper()})...")
        
        # Détecter le nombre de CPU disponibles et n_jobs optimal
        n_cpus = os.cpu_count() or 1
        n_jobs = get_optimal_n_jobs()
        
        if n_jobs == -1:
            print(f"  ⚙️ Paramètres : n_init={n_init}, max_iter=100, n_jobs=-1 (utilise {n_cpus} CPU)")
        else:
            print(f"  ⚙️ Paramètres : n_init={n_init}, max_iter=100, n_jobs={n_jobs} (Windows - mode sécurisé)")
        
        # Réduire n_init pour DTW si beaucoup de données
        if metric == "dtw" and len(self.scaled_curves) > 5000 and n_init > 5:
            n_init = 5
            print(f"  ⚡ DTW avec beaucoup de données → n_init réduit à {n_init}")
        
        # Essayer avec n_jobs, fallback sur 1 si erreur
        try:
            self.ts_kmeans_model = TimeSeriesKMeans(
                n_clusters=n_clusters, 
                metric=metric, 
                random_state=42, 
                n_init=n_init, 
                verbose=True,  # Afficher la progression
                max_iter=100,
                n_jobs=n_jobs
            )
            print("  🔄 Clustering en cours...")
            self.curves_df['cluster'] = self.ts_kmeans_model.fit_predict(self.scaled_curves)
        except Exception as e:
            # Si erreur avec parallélisme, fallback sur n_jobs=1
            if n_jobs != 1:
                print(f"  ⚠️ Erreur avec n_jobs={n_jobs}, fallback sur n_jobs=1...")
                self.ts_kmeans_model = TimeSeriesKMeans(
                    n_clusters=n_clusters, 
                    metric=metric, 
                    random_state=42, 
                    n_init=n_init, 
                    verbose=True,
                    max_iter=100,
                    n_jobs=1
                )
                print("  🔄 Clustering en cours (mode séquentiel)...")
                self.curves_df['cluster'] = self.ts_kmeans_model.fit_predict(self.scaled_curves)
            else:
                raise e
        
        # Afficher la distribution des clusters
        print("\n📊 Distribution des clusters :")
        cluster_counts = self.curves_df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.curves_df)) * 100
            print(f"  - Cluster {cluster_id}: {count} courbes ({percentage:.1f}%)")
        
        # Calculer les métriques de qualité (reshape pour sklearn)
        scaled_2d = self.scaled_curves.reshape(self.scaled_curves.shape[0], -1)
        silhouette = silhouette_score(scaled_2d, self.curves_df['cluster'])
        davies_bouldin = davies_bouldin_score(scaled_2d, self.curves_df['cluster'])
        
        print(f"\n📈 Métriques de qualité du clustering :")
        print(f"  - Score de Silhouette : {silhouette:.3f}")
        print(f"  - Davies-Bouldin Score : {davies_bouldin:.3f}")
        
        return self.curves_df
    
    def visualize_clusters(self):
        """
        Visualise les clusters de courbes de montée en charge.
        Les courbes montent de gauche à droite : J-30 (gauche) vers J (droite).
        """
        print("\n📊 Création des visualisations...")
        
        # feature_cols est dans l'ordre : ['J-30', 'J-29', ..., 'J-1', 'J-0']
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        n_clusters = self.curves_df['cluster'].nunique()
        
        # Figure 1 : Toutes les courbes par cluster (grille de 3 colonnes)
        n_cols = 3
        n_rows = (n_clusters + n_cols - 1) // n_cols  # Arrondi vers le haut
        fig_height = max(8, n_rows * 4)  # Ajuster la hauteur selon le nombre de lignes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
        
        # Aplatir les axes pour faciliter l'itération
        if n_clusters == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for cluster_id in range(n_clusters):
            cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id][feature_cols]
            
            ax = axes[cluster_id]
            
            # Tracer toutes les courbes du cluster en transparence
            for idx, row in cluster_data.iterrows():
                ax.plot(range(self.days_before + 1), row.values, 
                       alpha=0.1, color=f'C{cluster_id}')
            
            # Tracer la moyenne du cluster en gras
            mean_curve = cluster_data.mean()
            ax.plot(range(self.days_before + 1), mean_curve.values, 
                   color=f'C{cluster_id}', linewidth=3, 
                   label=f'Moyenne (n={len(cluster_data)})')
            
            # Tracer les percentiles
            q25 = cluster_data.quantile(0.25)
            q75 = cluster_data.quantile(0.75)
            ax.fill_between(range(self.days_before + 1), q25.values, q75.values,
                           alpha=0.2, color=f'C{cluster_id}')
            
            ax.set_title(f'Cluster {cluster_id} - {len(cluster_data)} courbes', 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel('Jours avant séjour', fontsize=8)
            ax.set_ylabel('Taux d\'occupation', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Personnaliser les labels de l'axe x (de J-30 à J)
            xticks_pos = [0, self.days_before // 2, self.days_before]
            xticks_labels = [f'J-{self.days_before}', f'J-{self.days_before // 2}', 'J']
            ax.set_xticks(xticks_pos)
            ax.set_xticklabels(xticks_labels)
        
        # Supprimer les axes non utilisés
        for idx in range(n_clusters, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        output_path = f'{self.results_dir}/clustering_curves_by_cluster.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'clustering_curves_by_cluster.png', 'Graphique courbes par cluster')
        
        plt.show()
        
        # Figure 2 : Comparaison des moyennes de tous les clusters
        plt.figure(figsize=(14, 8))
        
        for cluster_id in range(n_clusters):
            cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id][feature_cols]
            mean_curve = cluster_data.mean()
            std_curve = cluster_data.std()
            
            x_values = range(self.days_before + 1)
            plt.plot(x_values, mean_curve.values, linewidth=3, 
                    label=f'Cluster {cluster_id} (n={len(cluster_data)})', 
                    marker='o', markersize=4)
            
            # Ajouter l'intervalle de confiance
            plt.fill_between(x_values, 
                           mean_curve.values - std_curve.values,
                           mean_curve.values + std_curve.values,
                           alpha=0.15)
        
        plt.xlabel('Jours avant séjour', fontsize=9)
        plt.ylabel('Taux d\'occupation (moyenne)', fontsize=9)
        plt.title('Comparaison des profils de montée en charge par cluster', 
                 fontsize=10, fontweight='bold')
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Personnaliser les labels de l'axe x (dynamique selon days_before)
        step = max(5, self.days_before // 6)  # Environ 6-7 labels
        xticks_pos = list(range(0, self.days_before + 1, step))
        if xticks_pos[-1] != self.days_before:
            xticks_pos.append(self.days_before)
        xticks_labels = [f'J-{self.days_before - pos}' if pos < self.days_before else 'J' 
                        for pos in xticks_pos]
        plt.xticks(xticks_pos, xticks_labels)
        
        plt.tight_layout()
        output_path = f'{self.results_dir}/clustering_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'clustering_comparison.png', 'Graphique comparaison clusters')
        
        plt.show()
        
        # Figure 3 : Visualisation PCA en 2D
        print("\n🔬 Projection PCA en 2D...")
        pca = PCA(n_components=2)
        # Convertir en 2D pour PCA
        scaled_2d = self.scaled_curves.reshape(self.scaled_curves.shape[0], -1)
        pca_coords = pca.fit_transform(scaled_2d)
        
        plt.figure(figsize=(12, 8))
        
        for cluster_id in range(n_clusters):
            mask = self.curves_df['cluster'] == cluster_id
            plt.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                       alpha=0.6, s=50, label=f'Cluster {cluster_id}',
                       edgecolors='black', linewidth=0.5)
        
        # Tracer les centres des clusters
        # Convertir les centres en 2D pour PCA
        centers_2d = self.ts_kmeans_model.cluster_centers_.reshape(self.ts_kmeans_model.cluster_centers_.shape[0], -1)
        centers_pca = pca.transform(centers_2d)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='X', s=500, edgecolors='black', 
                   linewidth=2, label='Centres', zorder=10)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
        plt.title('Projection PCA des clusters de montée en charge', 
                 fontsize=10, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.results_dir}/clustering_pca.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'clustering_pca.png', 'Graphique PCA')
        
        plt.show()
        
        print(f"  - Variance expliquée (PC1+PC2) : "
              f"{sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    def analyze_cluster_characteristics(self):
        """
        Analyse les caractéristiques de chaque cluster.
        """
        print("\n📋 Analyse des caractéristiques des clusters :\n")
        
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        n_clusters = self.curves_df['cluster'].nunique()
        
        cluster_summary = []
        
        for cluster_id in range(n_clusters):
            cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id]
            cluster_curves = cluster_data[feature_cols]
            
            print(f"{'='*60}")
            print(f"CLUSTER {cluster_id} - {len(cluster_data)} courbes ({len(cluster_data)/len(self.curves_df)*100:.1f}%)")
            print(f"{'='*60}")
            
            # Statistiques de base
            final_to = cluster_curves['J-0']
            initial_to = cluster_curves[f'J-{self.days_before}']
            
            print(f"\n📊 Taux d'occupation à J :")
            print(f"  - Moyenne : {final_to.mean():.3f} ({final_to.mean()*100:.1f}%)")
            print(f"  - Médiane : {final_to.median():.3f} ({final_to.median()*100:.1f}%)")
            print(f"  - Écart-type : {final_to.std():.3f}")
            print(f"  - Min/Max : {final_to.min():.3f} / {final_to.max():.3f}")
            
            # Calcul de la vitesse de montée en charge
            growth = final_to - initial_to
            growth_rate = (growth / (initial_to + 0.01)) * 100  # +0.01 pour éviter division par 0
            
            print(f"\n📈 Croissance (J-{self.days_before} à J) :")
            print(f"  - Croissance moyenne : {growth.mean():.3f} ({growth.mean()*100:.1f} points de %)")
            print(f"  - Taux de croissance moyen : {growth_rate.mean():.1f}%")
            
            # Analyser la forme de la courbe
            mean_curve = cluster_curves.mean()
            
            # Calculer la pente moyenne sur différentes périodes (diviser en 3 tiers)
            third = len(mean_curve) // 3
            two_thirds = 2 * third
            
            early_period = mean_curve.iloc[:third].values  # Premier tiers
            mid_period = mean_curve.iloc[third:two_thirds].values   # Deuxième tiers
            late_period = mean_curve.iloc[two_thirds:].values    # Dernier tiers
            
            early_slope = np.mean(np.diff(early_period))
            mid_slope = np.mean(np.diff(mid_period))
            late_slope = np.mean(np.diff(late_period))
            
            # Calculer les jours correspondants aux périodes
            early_days = f"J-{self.days_before} à J-{self.days_before - third}"
            mid_days = f"J-{self.days_before - third} à J-{self.days_before - two_thirds}"
            late_days = f"J-{self.days_before - two_thirds} à J"
            
            print(f"\n🔍 Dynamique de la montée en charge :")
            print(f"  - Pente {early_days} : {early_slope:.4f} (To/jour)")
            print(f"  - Pente {mid_days} : {mid_slope:.4f} (To/jour)")
            print(f"  - Pente {late_days} : {late_slope:.4f} (To/jour)")
            
            # Interpréter le profil
            print(f"\n💡 Profil de réservation :")
            if late_slope > mid_slope and late_slope > early_slope:
                profile_type = "Dernière minute"
                print("  → Montée en charge accélérée en fin de période (réservations de dernière minute)")
            elif early_slope > mid_slope and early_slope > late_slope:
                profile_type = "Anticipé"
                print("  → Montée en charge rapide au début puis stabilisation (réservations anticipées)")
            elif abs(early_slope - late_slope) < 0.005 and mid_slope > 0:
                profile_type = "Régulier"
                print("  → Montée en charge régulière tout au long de la période")
            elif final_to.mean() < 0.2:
                profile_type = "Faible demande"
                print("  → Faible niveau de réservations (taux d'occupation < 20%)")
            else:
                profile_type = "Mixte"
                print("  → Profil mixte avec variations")
            
            # Dates de séjour caractéristiques
            print(f"\n📅 Période des séjours :")
            print(f"  - Première date : {cluster_data['stay_date'].min().strftime('%Y-%m-%d')}")
            print(f"  - Dernière date : {cluster_data['stay_date'].max().strftime('%Y-%m-%d')}")
            
            # Analyser la saisonnalité
            cluster_data['month'] = pd.to_datetime(cluster_data['stay_date']).dt.month
            month_dist = cluster_data['month'].value_counts().sort_index()
            top_months = month_dist.nlargest(3)
            month_names = {1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Jun',
                          7: 'Jul', 8: 'Aoû', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'}
            
            print(f"\n🗓️  Top 3 mois de séjour :")
            for month, count in top_months.items():
                print(f"  - {month_names[month]} : {count} dates ({count/len(cluster_data)*100:.1f}%)")
            
            cluster_summary.append({
                'cluster': cluster_id,
                'n_curves': len(cluster_data),
                'percentage': f"{len(cluster_data)/len(self.curves_df)*100:.1f}%",
                'avg_final_to': f"{final_to.mean():.2f} ({final_to.mean()*100:.1f}%)",
                'profile_type': profile_type,
                'late_slope': f"{late_slope:.4f}",
                'early_slope': f"{early_slope:.4f}"
            })
            
            print()
        
        # Créer un tableau récapitulatif
        summary_df = pd.DataFrame(cluster_summary)
        print(f"\n{'='*80}")
        print("TABLEAU RÉCAPITULATIF DES CLUSTERS")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def save_model(self, model_path=None):
        """
        Sauvegarde le modèle de clustering et le scaler.
        
        Args:
            model_path (str, optional): Chemin du fichier modèle. Si None, utilise results/{hotCode}/clustering_model.pkl
        """
        if model_path is None:
            model_path = f'{self.results_dir}/clustering_model.pkl'
        
        print(f"\n💾 Sauvegarde du modèle de clustering...")
        
        model_data = {
            'ts_kmeans_model': self.ts_kmeans_model,
            'scaler': self.scaler,
            'optimal_k': self.optimal_k,
            'days_before': self.days_before,
            'hotCode': self.hotCode
        }
        
        # Sauvegarder localement d'abord
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Puis vers Azure en priorité
        self._save_file(model_path, 'clustering_model.pkl', 'Modèle de clustering')
        
        return model_path
    
    def load_model(self, model_path=None):
        """
        Charge un modèle de clustering sauvegardé.
        
        Args:
            model_path (str, optional): Chemin du fichier modèle. Si None, utilise results/{hotCode}/clustering_model.pkl
        """
        import pickle
        
        if model_path is None:
            model_path = f'{self.results_dir}/clustering_model.pkl'
        
        print(f"\n📂 Chargement du modèle de clustering...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ts_kmeans_model = model_data['ts_kmeans_model']
        self.scaler = model_data['scaler']
        self.optimal_k = model_data['optimal_k']
        self.days_before = model_data['days_before']
        
        # Charger hotCode si disponible (rétrocompatibilité)
        if 'hotCode' in model_data:
            self.hotCode = model_data['hotCode']
        
        print(f"✓ Modèle chargé : {model_path}")
        print(f"  - Nombre de clusters : {self.optimal_k}")
        print(f"  - Jours avant séjour : {self.days_before}")
        if self.hotCode:
            print(f"  - Code hôtel : {self.hotCode}")
        return self
    
    def save_cluster_profiles(self, output_path=None):
        """
        Sauvegarde les profils moyens de chaque cluster dans un fichier CSV.
        Chaque ligne représente le profil moyen d'un cluster.
        
        Args:
            output_path (str, optional): Chemin du fichier de sortie. Si None, utilise results/{hotCode}/cluster_profiles.csv
        """
        if output_path is None:
            output_path = f'{self.results_dir}/cluster_profiles.csv'
        
        print(f"\n💾 Sauvegarde des profils moyens des clusters...")
        
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        n_clusters = self.curves_df['cluster'].nunique()
        
        profiles = []
        
        for cluster_id in range(n_clusters):
            cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id]
            cluster_curves = cluster_data[feature_cols]
            
            # Calculer le profil moyen
            mean_profile = cluster_curves.mean()
            std_profile = cluster_curves.std()
            
            # Statistiques supplémentaires
            profile_dict = {
                'cluster': cluster_id,
                'n_samples': len(cluster_data),
                'percentage': len(cluster_data) / len(self.curves_df) * 100
            }
            
            # Ajouter les valeurs moyennes pour chaque jour
            for col in feature_cols:
                profile_dict[f'{col}_mean'] = mean_profile[col]
                profile_dict[f'{col}_std'] = std_profile[col]
            
            profiles.append(profile_dict)
        
        profiles_df = pd.DataFrame(profiles)
        profiles_df.to_csv(output_path, index=False, sep=';')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'cluster_profiles.csv', 'Profils moyens des clusters')
        
        print(f"  - {len(profiles_df)} profils (un par cluster)")
        print(f"  - Colonnes : cluster, n_samples, percentage, J-{self.days_before}_mean/std ... J-0_mean/std")
        
        return profiles_df
    
    def get_cluster_profile(self, cluster_id):
        """
        Retourne le profil moyen d'un cluster spécifique.
        
        Args:
            cluster_id (int): ID du cluster
            
        Returns:
            dict: Dictionnaire avec le profil moyen et les statistiques
        """
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        
        cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            return None
        
        cluster_curves = cluster_data[feature_cols]
        
        profile = {
            'cluster': cluster_id,
            'n_samples': len(cluster_data),
            'mean_curve': cluster_curves.mean().to_dict(),
            'std_curve': cluster_curves.std().to_dict(),
            'median_curve': cluster_curves.median().to_dict(),
            'q25_curve': cluster_curves.quantile(0.25).to_dict(),
            'q75_curve': cluster_curves.quantile(0.75).to_dict()
        }
        
        return profile
    
    def predict_cluster(self, partial_curve, days_available=None):
        """
        Prédit le cluster pour une courbe incomplète (montée en charge partielle).
        
        Args:
            partial_curve (dict ou pd.Series): Courbe partielle avec les jours disponibles.
                                               Format: {'J-30': 0.1, 'J-29': 0.12, ..., 'J-15': 0.25}
            days_available (int, optional): Nombre de jours disponibles depuis J-days_before.
                                           Si None, détecté automatiquement depuis partial_curve.
        
        Returns:
            dict: Informations sur le cluster prédit
                  - 'cluster': ID du cluster prédit
                  - 'confidence': Score de confiance (distance au centre le plus proche)
                  - 'all_distances': Distances à tous les centres de clusters
        """
        print(f"\n🔮 Prédiction du cluster pour une courbe incomplète...")
        
        feature_cols = [f'J-{i}' for i in range(self.days_before, -1, -1)]
        
        # Convertir en Series si dict
        if isinstance(partial_curve, dict):
            partial_curve = pd.Series(partial_curve)
        
        # Créer une courbe complète avec interpolation/extrapolation
        full_curve = pd.Series(index=feature_cols, dtype=float)
        
        # Remplir les valeurs disponibles
        for col in feature_cols:
            if col in partial_curve.index and not pd.isna(partial_curve[col]):
                full_curve[col] = partial_curve[col]
            else:
                full_curve[col] = np.nan
        
        # Déterminer jusqu'où on a des données
        last_valid_idx = full_curve.last_valid_index()
        if last_valid_idx is None:
            raise ValueError("Aucune donnée valide dans la courbe partielle")
        
        last_valid_pos = feature_cols.index(last_valid_idx)
        print(f"  - Données disponibles jusqu'à {last_valid_idx} ({last_valid_pos + 1}/{len(feature_cols)} jours)")
        
        # Interpoler les valeurs manquantes au début/milieu
        full_curve = full_curve.interpolate(method='linear', limit_direction='both')
        
        # Pour les valeurs futures (après last_valid_idx), utiliser les profils moyens des clusters
        # Stratégie : tester chaque cluster et voir lequel correspond le mieux
        
        if pd.isna(full_curve.iloc[-1]):  # Si on n'a pas J-0
            print(f"  - Extrapolation nécessaire pour les jours futurs")
            
            # Calculer la distance de la courbe partielle à chaque cluster
            best_cluster = None
            best_distance = float('inf')
            all_distances = {}
            
            for cluster_id in range(self.ts_kmeans_model.n_clusters):
                # Obtenir le profil moyen du cluster
                cluster_data = self.curves_df[self.curves_df['cluster'] == cluster_id]
                cluster_mean = cluster_data[feature_cols].mean()
                
                # Calculer la distance seulement sur les jours disponibles
                partial_cols = feature_cols[:last_valid_pos + 1]
                distance = np.sqrt(np.sum((full_curve[partial_cols] - cluster_mean[partial_cols]) ** 2))
                all_distances[cluster_id] = distance
                
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster_id
            
            # Compléter avec le profil du cluster le plus proche
            cluster_data = self.curves_df[self.curves_df['cluster'] == best_cluster]
            cluster_mean = cluster_data[feature_cols].mean()
            
            # Extrapoler les valeurs futures
            for i in range(last_valid_pos + 1, len(feature_cols)):
                full_curve.iloc[i] = cluster_mean.iloc[i]
            
            print(f"  - Extrapolation effectuée avec le profil du cluster {best_cluster}")
        
        # Normaliser la courbe complète
        curve_3d = full_curve.values[:, np.newaxis, np.newaxis].T
        scaled_curve = self.scaler.transform(curve_3d)
        
        # Prédire le cluster
        predicted_cluster = self.ts_kmeans_model.predict(scaled_curve)[0]
        
        # Calculer les distances à tous les centres
        centers = self.ts_kmeans_model.cluster_centers_
        distances = {}
        for cluster_id in range(len(centers)):
            dist = np.linalg.norm(scaled_curve - centers[cluster_id])
            distances[cluster_id] = dist
        
        # Confidence = 1 / (1 + distance_min)
        min_distance = min(distances.values())
        confidence = 1 / (1 + min_distance)
        
        print(f"\n✓ Cluster prédit : {predicted_cluster}")
        print(f"  - Confiance : {confidence:.3f}")
        print(f"  - Distance au centre : {min_distance:.3f}")
        
        # Afficher les distances à tous les clusters
        print(f"\n  📊 Distances à tous les clusters :")
        for cluster_id in sorted(distances.keys()):
            dist = distances[cluster_id]
            marker = "👉" if cluster_id == predicted_cluster else "  "
            print(f"    {marker} Cluster {cluster_id}: {dist:.3f}")
        
        return {
            'cluster': predicted_cluster,
            'confidence': confidence,
            'all_distances': distances,
            'full_curve': full_curve.to_dict()
        }
    
    def save_results(self, output_path=None):
        """
        Sauvegarde les résultats du clustering dans un fichier CSV.
        
        Args:
            output_path (str, optional): Chemin du fichier de sortie. Si None, utilise results/{hotCode}/clustering_results.csv
        """
        if output_path is None:
            output_path = f'{self.results_dir}/clustering_results.csv'
        
        print(f"\n💾 Sauvegarde des résultats...")
        
        # Créer un DataFrame avec toutes les informations
        results_df = self.curves_df.copy()
        
        # Sauvegarder localement
        results_df.to_csv(output_path, index=False, sep=';')
        
        # Sauvegarder vers Azure en priorité
        self._save_file(output_path, 'clustering_results.csv', 'Résultats du clustering')
        
        print(f"  - {len(results_df)} courbes avec leur cluster assigné")
        
        return results_df


def main(hotCode=None):
    """
    Fonction principale pour exécuter l'analyse de clustering.
    
    Args:
        hotCode (str, optional): Code de l'hôtel (3 caractères, ex: 'D09'). 
                                 Si None, lit depuis les arguments de ligne de commande.
    
    Usage:
        python prediction_cluster.py D09
    """
    import sys
    
    print("="*80)
    print("ANALYSE DE CLUSTERING - MONTÉE EN CHARGE DES RÉSERVATIONS HÔTELIÈRES")
    print("="*80)
    print()
    
    # Lire le code hôtel depuis les arguments si non fourni
    if hotCode is None:
        if len(sys.argv) < 2:
            print("❌ ERREUR : Code hôtel manquant !")
            print()
            print("Usage:")
            print(f"  python {sys.argv[0]} <hotCode>")
            print()
            print("Exemple:")
            print(f"  python {sys.argv[0]} D09")
            print()
            sys.exit(1)
        
        hotCode = sys.argv[1].strip().upper()
    
    # Validation du code hôtel
    if len(hotCode) != 3:
        print(f"⚠️ ATTENTION : Le code '{hotCode}' ne fait pas 3 caractères")
        print("   Le code hôtel devrait être de 3 caractères (ex: D09, A12, B05)")
        print()
    
    print(f"🏨 Hôtel sélectionné : {hotCode}")
    print(f"📂 Données : data/{hotCode}/Indicateurs.csv")
    print(f"💾 Résultats : results/{hotCode}/ (local)")
    
    # Information sur le chargement des variables d'environnement
    if DOTENV_AVAILABLE:
        from pathlib import Path
        env_file = Path('.env')
        if env_file.exists():
            print("✅ Fichier .env détecté et chargé")
        else:
            print("ℹ️  Fichier .env non trouvé (utilisation des variables d'environnement système)")
    else:
        print("ℹ️  Module python-dotenv non disponible (utilisation des variables d'environnement système uniquement)")
    
    if AZURE_AVAILABLE:
        if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            print("☁️  Azure Blob Storage : Configuration détectée")
        else:
            print("⚠️  Azure Blob Storage : AZURE_STORAGE_CONNECTION_STRING non définie")
    else:
        print("⚠️  Azure Blob Storage : SDK non disponible (installer azure-storage-blob)")
    
    print()
    
    # Configuration
    DAYS_BEFORE = 60  # Nombre de jours avant le séjour à analyser (J-60 à J)
    YEAR_FILTER = None  # Filtrer sur l'année 2024 (None = toutes les années)
    
    # Options de lissage (réduction du bruit)
    ENABLE_SMOOTHING = True  # Activer le lissage Savitzky-Golay
    SMOOTHING_WINDOW = 15  # Longueur de la fenêtre (doit être impair) - Augmenté pour plus de lissage
    SMOOTHING_POLYORDER = 3  # Ordre du polynôme - Augmenté pour courbe plus lisse
    SMOOTHING_DOUBLE_PASS = False  # Double passage pour lissage renforcé (True = encore plus lisse)
    
    # Options de clustering
    N_CLUSTERS = 5  # Nombre de clusters (par défaut : 12)
    AUTO_FIND_K = False  # Recherche automatique du nombre optimal de clusters (True pour activer)
    USE_DTW = True  # True = DTW (meilleure qualité, lent) | False = Euclidean (rapide, dev)
    
    # Initialiser et exécuter l'analyse
    clustering = HotelBookingClustering(hotCode=hotCode, days_before=DAYS_BEFORE)
    
    # Étape 1 : Charger les données (filtrées sur 2024)
    clustering.load_data(year_filter=YEAR_FILTER)
    
    # Étape 2 : Préparer les courbes de montée en charge
    clustering.prepare_booking_curves(min_observations=20)
    
    # Étape 2.5 : Appliquer le lissage pour réduire le bruit (optionnel)
    clustering.apply_smoothing(
        enable=ENABLE_SMOOTHING,
        window_length=SMOOTHING_WINDOW,
        polyorder=SMOOTHING_POLYORDER,
        double_pass=SMOOTHING_DOUBLE_PASS
    )
    
    # Étape 3 : Analyser les taux d'occupation initiaux (à J-30)
    clustering.analyze_initial_occupancy()
    
    # Étape 4 : Normaliser les données
    clustering.normalize_curves()
    
    # Étape 5 : Détermination du nombre de clusters
    if AUTO_FIND_K:
        # Recherche automatique activée
        print("\n" + "="*80)
        print("💡 ÉTAPE 5 : Recherche du nombre optimal de clusters")
        print("  - Métrique : euclidean (rapide)")
        print("  - Plage : K=2 à K=10")
        print("="*80)
        search_metric = "euclidean"
        optimal_k = clustering.find_optimal_clusters(max_k=10, metric=search_metric)
        print(f"\n✓ K optimal suggéré : {optimal_k}")
    else:
        # Utiliser le nombre de clusters configuré
        optimal_k = N_CLUSTERS
        print("\n" + "="*80)
        print("💡 ÉTAPE 5 : Configuration du clustering")
        print(f"  - Nombre de clusters : {optimal_k} (configuré)")
        print("  - Recherche automatique : DÉSACTIVÉE")
        print(f"  - Pour activer : AUTO_FIND_K = True")
        print("="*80)
    
    # Étape 6 : Effectuer le clustering final
    if USE_DTW:
        final_metric = "dtw"
        final_n_init = 5
        mode_text = "DTW (meilleure qualité)"
    else:
        final_metric = "euclidean"
        final_n_init = 10
        mode_text = "EUCLIDEAN (rapide - mode développement)"
    
    print("\n" + "="*80)
    print("💡 ÉTAPE 6 : Clustering final")
    print(f"  - Nombre de clusters : {optimal_k}")
    print(f"  - Métrique : {mode_text}")
    print(f"  - Initialisations : {final_n_init}")
    if not USE_DTW:
        print("  ⚠️  Mode développement - Changez USE_DTW = True pour la production")
    print("="*80)
    
    clustering.perform_clustering(n_clusters=optimal_k, metric=final_metric, n_init=final_n_init)
    
    # Étape 7 : Visualiser les clusters
    clustering.visualize_clusters()
    
    # Étape 8 : Analyser les caractéristiques
    summary = clustering.analyze_cluster_characteristics()
    
    # Étape 9 : Sauvegarder les résultats
    clustering.save_results()
    
    # Étape 10 : Sauvegarder le modèle de clustering
    clustering.save_model()
    
    # Étape 11 : Sauvegarder les profils moyens des clusters
    clustering.save_cluster_profiles()
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS !")
    print("="*80)
    
    # Afficher l'emplacement de sauvegarde
    if clustering.azure_enabled:
        print(f"\n☁️  Fichiers sauvegardés dans Azure Blob Storage:")
        print(f"  📦 Container : {clustering.azure_container}")
        print(f"  📁 Dossier : clustering/{hotCode}/")
        print(f"\n  Fichiers:")
        print(f"  - initial_occupancy_analysis.png (Analyse exploratoire)")
        print(f"  - clustering_curves_by_cluster.png")
        print(f"  - clustering_comparison.png")
        print(f"  - clustering_pca.png")
        print(f"  - clustering_results.csv")
        print(f"  - clustering_model.pkl (Modèle de clustering)")
        print(f"  - cluster_profiles.csv (Profils moyens des clusters)")
        if AUTO_FIND_K:
            print(f"  - clustering_optimal_k.png (Recherche K optimal)")
        print(f"\n  💾 Copie locale également disponible dans : results/{hotCode}/")
    else:
        print(f"\n📁 Fichiers générés dans results/{hotCode}/ :")
        print(f"  - results/{hotCode}/initial_occupancy_analysis.png (Analyse exploratoire)")
        print(f"  - results/{hotCode}/clustering_curves_by_cluster.png")
        print(f"  - results/{hotCode}/clustering_comparison.png")
        print(f"  - results/{hotCode}/clustering_pca.png")
        print(f"  - results/{hotCode}/clustering_results.csv")
        print(f"  - results/{hotCode}/clustering_model.pkl (Modèle de clustering)")
        print(f"  - results/{hotCode}/cluster_profiles.csv (Profils moyens des clusters)")
        if AUTO_FIND_K:
            print(f"  - results/{hotCode}/clustering_optimal_k.png (Recherche K optimal)")
    
    print("\n💡 UTILISATION POUR LA PRÉDICTION :")
    print(f"  1. Créer l'instance : clustering = HotelBookingClustering(hotCode='{hotCode}')")
    print(f"  2. Charger le modèle : clustering.load_model()")
    print("  3. Prédire un cluster : clustering.predict_cluster({'J-60': 0.1, 'J-59': 0.12, ...})")
    print("  4. Obtenir un profil : clustering.get_cluster_profile(cluster_id)")
    print()


if __name__ == "__main__":
    main()

