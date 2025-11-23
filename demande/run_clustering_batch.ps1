# Script PowerShell pour analyser plusieurs hôtels en batch
# Usage: .\run_clustering_batch.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  ANALYSE DE CLUSTERING EN BATCH - PLUSIEURS HÔTELS" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Liste des codes hôtel à analyser
$hotels = @("D09", "A12", "B05", "C23")

# Vérifier si Python est disponible
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python détecté : $pythonVersion" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "❌ ERREUR : Python n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    exit 1
}

# Compteurs
$totalHotels = $hotels.Count
$successCount = 0
$failedCount = 0
$failedHotels = @()

Write-Host "📋 Hôtels à traiter : $totalHotels" -ForegroundColor Yellow
Write-Host "    $($hotels -join ', ')" -ForegroundColor Yellow
Write-Host ""

# Traiter chaque hôtel
$currentIndex = 0
foreach ($hotel in $hotels) {
    $currentIndex++
    
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "  [$currentIndex/$totalHotels] Traitement de l'hôtel : $hotel" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Vérifier si le fichier de données existe
    $dataFile = "data\$hotel\Indicateurs.csv"
    if (-not (Test-Path $dataFile)) {
        Write-Host "⚠️  ATTENTION : Fichier non trouvé : $dataFile" -ForegroundColor Yellow
        Write-Host "    → Hôtel $hotel ignoré" -ForegroundColor Yellow
        Write-Host ""
        $failedCount++
        $failedHotels += $hotel
        continue
    }
    
    # Exécuter l'analyse
    $startTime = Get-Date
    Write-Host "🚀 Démarrage de l'analyse pour $hotel..." -ForegroundColor Green
    Write-Host ""
    
    try {
        python prediction_cluster.py $hotel
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            Write-Host ""
            Write-Host "✓ Hôtel $hotel terminé avec succès !" -ForegroundColor Green
            Write-Host "  Durée : $($duration.ToString('mm\:ss'))" -ForegroundColor Green
            Write-Host ""
            $successCount++
        } else {
            Write-Host ""
            Write-Host "❌ Erreur lors du traitement de l'hôtel $hotel" -ForegroundColor Red
            Write-Host ""
            $failedCount++
            $failedHotels += $hotel
        }
    } catch {
        Write-Host ""
        Write-Host "❌ Exception lors du traitement de l'hôtel $hotel : $_" -ForegroundColor Red
        Write-Host ""
        $failedCount++
        $failedHotels += $hotel
    }
}

# Résumé final
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  RÉSUMÉ DE L'ANALYSE EN BATCH" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total d'hôtels traités : $totalHotels" -ForegroundColor White
Write-Host "✓ Succès : $successCount" -ForegroundColor Green
Write-Host "❌ Échecs : $failedCount" -ForegroundColor Red

if ($failedCount -gt 0) {
    Write-Host ""
    Write-Host "Hôtels en échec : $($failedHotels -join ', ')" -ForegroundColor Red
}

Write-Host ""
Write-Host "📁 Les résultats sont disponibles dans : results\" -ForegroundColor Yellow
Write-Host ""

# Afficher les dossiers de résultats créés
Write-Host "Dossiers de résultats créés :" -ForegroundColor Yellow
foreach ($hotel in $hotels) {
    $resultDir = "results\$hotel"
    if (Test-Path $resultDir) {
        $fileCount = (Get-ChildItem -Path $resultDir -File).Count
        Write-Host "  ✓ $resultDir ($fileCount fichiers)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Code de sortie
if ($failedCount -gt 0) {
    exit 1
} else {
    exit 0
}

