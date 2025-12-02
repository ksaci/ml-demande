#!/bin/bash
# Script Linux/macOS pour compiler et exécuter l'analyse de saisonnalité

echo "========================================"
echo "Analyse de Saisonnalite - Version C#"
echo "========================================"
echo ""

# Vérifier si .NET SDK est installé
if ! command -v dotnet &> /dev/null; then
    echo "ERREUR: .NET SDK n'est pas installé"
    echo "Installez-le depuis: https://dotnet.microsoft.com/download"
    exit 1
fi

echo "[1/3] Restauration des packages NuGet..."
dotnet restore
if [ $? -ne 0 ]; then
    echo "ERREUR lors de la restauration des packages"
    exit 1
fi

echo ""
echo "[2/3] Compilation du projet..."
dotnet build -c Release
if [ $? -ne 0 ]; then
    echo "ERREUR lors de la compilation"
    exit 1
fi

echo ""
echo "[3/3] Exécution de l'analyse..."
echo ""
dotnet run -c Release

echo ""
echo "========================================"
echo "Analyse terminée"
echo "========================================"

