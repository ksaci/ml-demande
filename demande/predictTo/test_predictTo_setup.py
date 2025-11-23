"""
Script de test pour valider l'installation et la configuration.
Exécutez ce script avant de lancer l'entraînement complet.

Usage:
    python test_xgboost_setup.py
"""

import sys
import os
from pathlib import Path

print("🔍 Vérification de l'environnement PredictTO...")
print("=" * 60)

# 1. Vérifier les imports
print("\n1️⃣ Vérification des bibliothèques...")
missing_packages = []

try:
    import pandas
    print("  ✅ pandas")
except ImportError:
    print("  ❌ pandas")
    missing_packages.append("pandas")

try:
    import numpy
    print("  ✅ numpy")
except ImportError:
    print("  ❌ numpy")
    missing_packages.append("numpy")

try:
    import sklearn
    print("  ✅ scikit-learn")
except ImportError:
    print("  ❌ scikit-learn")
    missing_packages.append("scikit-learn")

try:
    import xgboost
    print("  ✅ xgboost")
except ImportError:
    print("  ❌ xgboost")
    missing_packages.append("xgboost")

try:
    import joblib
    print("  ✅ joblib")
except ImportError:
    print("  ❌ joblib")
    missing_packages.append("joblib")

try:
    import matplotlib
    print("  ✅ matplotlib")
except ImportError:
    print("  ❌ matplotlib")
    missing_packages.append("matplotlib")

try:
    import seaborn
    print("  ✅ seaborn")
except ImportError:
    print("  ❌ seaborn")
    missing_packages.append("seaborn")

try:
    from azure.storage.blob import BlobServiceClient
    print("  ✅ azure-storage-blob")
except ImportError:
    print("  ⚠️  azure-storage-blob (optionnel)")

if missing_packages:
    print(f"\n❌ Packages manquants: {', '.join(missing_packages)}")
    print(f"   Installez-les avec: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# 2. Vérifier les fichiers de données
print("\n2️⃣ Vérification des fichiers de données...")

data_files = [
    "../results/clustering_results.csv",
    "../data/Indicateurs.csv"
]

missing_files = []
for file_path in data_files:
    if Path(file_path).exists():
        file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
        print(f"  ✅ {file_path} ({file_size:.2f} MB)")
    else:
        print(f"  ❌ {file_path}")
        missing_files.append(file_path)

if missing_files:
    print(f"\n⚠️  Fichiers manquants: {', '.join(missing_files)}")
    print("   Le script ne pourra pas s'exécuter sans ces fichiers.")

# 3. Vérifier la configuration Azure
print("\n3️⃣ Vérification de la configuration Azure...")

azure_conn_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if azure_conn_string:
    print("  ✅ AZURE_STORAGE_CONNECTION_STRING définie")
    print("     La sauvegarde Azure sera activée")
else:
    print("  ⚠️  AZURE_STORAGE_CONNECTION_STRING non définie")
    print("     La sauvegarde Azure sera ignorée (sauvegarde locale uniquement)")

# 4. Vérifier les répertoires de sortie
print("\n4️⃣ Vérification des répertoires de sortie...")

output_dirs = [
    "/results/models",
    "/results"
]

for dir_path in output_dirs:
    if Path(dir_path).exists():
        print(f"  ✅ {dir_path}/")
    else:
        print(f"  📁 {dir_path}/ (sera créé automatiquement)")

# 5. Test d'import du module principal
print("\n5️⃣ Test d'import du module...")

try:
    from predictTo_train_model import XGBoostOccupancyPredictor
    print("  ✅ predictTo_train_model importé avec succès")
except ImportError as e:
    print(f"  ❌ Erreur d'import: {e}")
    sys.exit(1)

# Résumé
print("\n" + "=" * 60)
if missing_packages or missing_files:
    print("⚠️  CONFIGURATION INCOMPLÈTE")
    print("   Veuillez résoudre les problèmes ci-dessus avant de continuer.")
    sys.exit(1)
else:
    print("✅ CONFIGURATION VALIDE")
    print("   Vous pouvez lancer l'entraînement avec:")
    print("   python predictTo_train_model.py")
    print("=" * 60)

