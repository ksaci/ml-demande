# Script PowerShell pour créer le fichier .env
# Usage: .\create_env.ps1
# ATTENTION: Ce script crée un fichier .env VIDE - vous devez le remplir avec vos vraies clés Azure

$envFile = ".env"

if (Test-Path $envFile) {
    $replace = Read-Host "Le fichier .env existe déjà. Voulez-vous le remplacer? (o/n)"
    if ($replace -ne "o" -and $replace -ne "O") {
        Write-Host "Annulé"
        exit 0
    }
}

# Copier le fichier d'exemple
if (Test-Path ".env.example") {
    Copy-Item ".env.example" $envFile
    Write-Host "✅ Fichier .env créé depuis .env.example"
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Éditez le fichier .env et remplissez vos vraies clés Azure"
    Write-Host "    Les valeurs sont actuellement vides pour des raisons de sécurité"
    Write-Host ""
    Write-Host "Pour obtenir votre Connection String Azure:"
    Write-Host "1. Allez sur le portail Azure"
    Write-Host "2. Ouvrez votre Storage Account"
    Write-Host "3. Allez dans 'Clés d'accès'"
    Write-Host "4. Copiez la Connection String"
} else {
    Write-Host "❌ Erreur: .env.example n'existe pas"
    exit 1
}

