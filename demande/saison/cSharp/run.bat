@echo off
REM Script Windows pour compiler et exécuter l'analyse de saisonnalité

echo ========================================
echo Analyse de Saisonnalite - Version C#
echo ========================================
echo.

REM Vérifier si .NET SDK est installé
dotnet --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: .NET SDK n'est pas installé
    echo Installez-le depuis: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

echo [1/3] Restauration des packages NuGet...
dotnet restore
if %errorlevel% neq 0 (
    echo ERREUR lors de la restauration des packages
    pause
    exit /b 1
)

echo.
echo [2/3] Compilation du projet...
dotnet build -c Release
if %errorlevel% neq 0 (
    echo ERREUR lors de la compilation
    pause
    exit /b 1
)

echo.
echo [3/3] Exécution de l'analyse...
echo.
dotnet run -c Release

echo.
echo ========================================
echo Analyse terminée
echo ========================================
pause

