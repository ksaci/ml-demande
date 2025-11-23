"""
Script pour télécharger et utiliser un modèle PredictTO depuis Azure Blob Storage.

Ce script permet de :
1. Lister les modèles disponibles dans Azure
2. Télécharger un modèle spécifique
3. L'utiliser pour faire des prédictions

Usage:
    # Lister les modèles disponibles
    python load_predictTo_from_azure.py --list
    
    # Télécharger le dernier modèle
    python load_predictTo_from_azure.py --download latest
    
    # Télécharger un modèle spécifique
    python load_predictTo_from_azure.py --download 20241216_143025
"""

import os
import sys
import argparse
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


def get_blob_client(container_name: str = "prediction-demande"):
    """
    Crée un client Azure Blob Storage.
    
    Args:
        container_name: Nom du container
        
    Returns:
        ContainerClient
    """
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING non définie")
        print("   Définissez-la avec:")
        print('   export AZURE_STORAGE_CONNECTION_STRING="..."')
        sys.exit(1)
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    return container_client


def list_available_models(container_name: str = "prediction-demande"):
    """
    Liste les modèles disponibles dans Azure.
    
    Args:
        container_name: Nom du container
    """
    print(f"🔍 Recherche des modèles dans le container '{container_name}'...\n")
    
    try:
        container_client = get_blob_client(container_name)
        
        # Lister les blobs dans le dossier models/
        blob_list = container_client.list_blobs(name_starts_with="models/")
        
        models = {}
        for blob in blob_list:
            # Extraire le timestamp du chemin (models/TIMESTAMP/fichier.joblib)
            parts = blob.name.split('/')
            if len(parts) >= 3:
                timestamp = parts[1]
                if timestamp not in models:
                    models[timestamp] = []
                models[timestamp].append(parts[2])
        
        if not models:
            print("ℹ️  Aucun modèle trouvé dans Azure")
            return
        
        print(f"📦 {len(models)} version(s) de modèle trouvée(s):\n")
        
        for timestamp in sorted(models.keys(), reverse=True):
            print(f"  📅 {timestamp}")
            for filename in sorted(models[timestamp]):
                print(f"     - {filename}")
            print()
        
        latest = sorted(models.keys(), reverse=True)[0]
        print(f"💡 Dernier modèle: {latest}")
        print(f"   Pour le télécharger: python load_predictTo_from_azure.py --download {latest}")
        
    except ResourceNotFoundError:
        print(f"❌ Container '{container_name}' non trouvé")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)


def download_model(timestamp: str, container_name: str = "prediction-demande", output_dir: str = "models_azure"):
    """
    Télécharge un modèle depuis Azure.
    
    Args:
        timestamp: Timestamp du modèle (ex: '20241216_143025' ou 'latest')
        container_name: Nom du container
        output_dir: Répertoire de téléchargement local
    """
    print(f"⬇️  Téléchargement du modèle depuis Azure...\n")
    
    try:
        container_client = get_blob_client(container_name)
        
        # Si 'latest', trouver le dernier timestamp
        if timestamp == 'latest':
            blob_list = list(container_client.list_blobs(name_starts_with="models/"))
            if not blob_list:
                print("❌ Aucun modèle trouvé dans Azure")
                sys.exit(1)
            
            timestamps = set()
            for blob in blob_list:
                parts = blob.name.split('/')
                if len(parts) >= 2:
                    timestamps.add(parts[1])
            
            timestamp = sorted(timestamps, reverse=True)[0]
            print(f"📅 Dernier modèle trouvé: {timestamp}")
        
        # Créer le répertoire de sortie
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Télécharger les fichiers
        files_to_download = [
            "xgb_to_predictor.joblib",
            "xgb_scaler.joblib",
            "feature_columns.txt"
        ]
        
        for filename in files_to_download:
            blob_name = f"models/{timestamp}/{filename}"
            local_path = output_path / filename
            
            blob_client = container_client.get_blob_client(blob_name)
            
            print(f"  ⬇️  {filename}...", end=" ")
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            file_size = local_path.stat().st_size / 1024  # KB
            print(f"✅ ({file_size:.1f} KB)")
        
        print(f"\n✅ Modèle téléchargé avec succès dans: {output_path}")
        print(f"\n💡 Utilisez-le avec:")
        print(f"   from predictTo_predict_example import load_model_artifacts")
        print(f"   model, scaler, features = load_model_artifacts('{output_path}')")
        
    except ResourceNotFoundError:
        print(f"❌ Modèle '{timestamp}' non trouvé dans Azure")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement: {e}")
        sys.exit(1)


def main():
    """
    Point d'entrée principal.
    """
    parser = argparse.ArgumentParser(
        description="Gestion des modèles PredictTO dans Azure Blob Storage"
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Lister les modèles disponibles dans Azure'
    )
    
    parser.add_argument(
        '--download',
        type=str,
        metavar='TIMESTAMP',
        help="Télécharger un modèle (ex: '20241216_143025' ou 'latest')"
    )
    
    parser.add_argument(
        '--container',
        type=str,
        default='prediction-demande',
        help='Nom du container Azure (défaut: prediction-demande)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='models_azure',
        help='Répertoire de téléchargement local (défaut: models_azure)'
    )
    
    args = parser.parse_args()
    
    # Afficher l'aide si aucun argument
    if not args.list and not args.download:
        parser.print_help()
        sys.exit(0)
    
    # Lister les modèles
    if args.list:
        list_available_models(args.container)
    
    # Télécharger un modèle
    if args.download:
        download_model(args.download, args.container, args.output)


if __name__ == "__main__":
    main()

