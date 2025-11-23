"""
Script pour copier et renommer les fichiers CSV indicateurs.
Copie tous les fichiers CSV indicateurs des sous-dossiers de D:/Export_25_06_2025/hotcode
vers demande/data avec le format {hotCode}_indicateurs.csv
"""
import os
import shutil
from pathlib import Path

# Chemins
SOURCE_DIR = Path(r"D:\Export_25_06_2025")
DEST_DIR = Path(r"c:\github\machineLearning\demande\data")

# Créer le dossier de destination s'il n'existe pas
DEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Recherche des fichiers CSV indicateurs dans : {SOURCE_DIR}")
print(f"Destination : {DEST_DIR}\n")

# Compteurs
files_found = 0
files_copied = 0
errors = []

# Parcourir tous les sous-dossiers
if not SOURCE_DIR.exists():
    print(f"ERREUR : Le dossier source {SOURCE_DIR} n'existe pas !")
    exit(1)

# Chercher tous les fichiers CSV dans les sous-dossiers
for csv_file in SOURCE_DIR.rglob("*.csv"):
    files_found += 1
    file_name = csv_file.name.lower()
    
    # Vérifier si c'est un fichier indicateur (plusieurs variantes possibles)
    is_indicateur = (
        "indicateur" in file_name or
        "indicateurs" in file_name
    )
    
    if is_indicateur:
        # Le code hôtel est le nom du dossier parent direct où se trouve le fichier
        # Structure attendue : D:\Export_25_06_2025\ABC\indicateurs.csv
        # Le code hôtel est "ABC" (nom du dossier parent)
        hot_code = None
        
        # Prendre le nom du dossier parent direct
        parent_name = csv_file.parent.name
        
        # Vérifier que c'est un code de 3 caractères alphanumériques (code hôtel)
        # Les codes peuvent contenir des lettres ET des chiffres (ex: 0DX, 01S, ABC)
        if len(parent_name) == 3 and parent_name.isalnum():
            hot_code = parent_name.upper()
        else:
            # Si le parent n'est pas un code de 3 caractères, afficher un avertissement
            print(f"ATTENTION : Le dossier parent '{parent_name}' n'est pas un code hotel valide (3 caracteres alphanumeriques) pour : {csv_file}")
        
        if hot_code:
            # Créer le nouveau nom de fichier
            new_name = f"{hot_code}_indicateurs.csv"
            dest_path = DEST_DIR / new_name
            
            # Copier le fichier
            try:
                shutil.copy2(csv_file, dest_path)
                files_copied += 1
                print(f"OK - Copie : {csv_file.name} -> {new_name}")
            except Exception as e:
                error_msg = f"Erreur lors de la copie de {csv_file}: {e}"
                errors.append(error_msg)
                print(f"ERREUR : {error_msg}")
        else:
            print(f"ATTENTION : Code hotel non trouve pour : {csv_file}")
            errors.append(f"Code hotel non trouve : {csv_file}")

print(f"\n{'='*60}")
print(f"RESUME :")
print(f"   - Fichiers CSV trouves : {files_found}")
print(f"   - Fichiers indicateurs copies : {files_copied}")
print(f"   - Erreurs : {len(errors)}")

if errors:
    print(f"\nERREURS rencontrees :")
    for error in errors[:10]:  # Afficher les 10 premières erreurs
        print(f"   - {error}")

if files_copied > 0:
    print(f"\nSUCCES : {files_copied} fichier(s) copie(s) dans {DEST_DIR}")

