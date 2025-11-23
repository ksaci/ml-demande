#!/bin/bash
# Script pour créer le fichier .env
# Usage: bash create_env.sh
# ATTENTION: Ce script crée un fichier .env VIDE - vous devez le remplir avec vos vraies clés Azure

ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  Le fichier .env existe déjà"
    read -p "Voulez-vous le remplacer? (o/n): " replace
    if [ "$replace" != "o" ] && [ "$replace" != "O" ]; then
        echo "Annulé"
        exit 0
    fi
fi

# Copier le fichier d'exemple
if [ -f ".env.example" ]; then
    cp ".env.example" "$ENV_FILE"
    echo "✅ Fichier .env créé depuis .env.example"
    echo ""
    echo "⚠️  IMPORTANT: Éditez le fichier .env et remplissez vos vraies clés Azure"
    echo "    Les valeurs sont actuellement vides pour des raisons de sécurité"
    echo ""
    echo "Pour obtenir votre Connection String Azure:"
    echo "1. Allez sur le portail Azure"
    echo "2. Ouvrez votre Storage Account"
    echo "3. Allez dans 'Clés d'accès'"
    echo "4. Copiez la Connection String"
else
    echo "❌ Erreur: .env.example n'existe pas"
    exit 1
fi

