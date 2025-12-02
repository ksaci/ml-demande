"""
Script de test pour vérifier le calcul du R².

Ce script permet de tester que le R² est calculé correctement
et détecte les problèmes d'affichage ou de calcul.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 80)
print("TEST DU CALCUL DU R²")
print("=" * 80)

# Test 1: Prédictions parfaites (R² = 1.0)
print("\n1️⃣  Test avec prédictions parfaites:")
y_true_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y_pred_1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
r2_1 = r2_score(y_true_1, y_pred_1)
mae_1 = mean_absolute_error(y_true_1, y_pred_1)
print(f"   R² = {r2_1:.4f} (attendu: 1.0000)")
print(f"   MAE = {mae_1:.4f} (attendu: 0.0000)")
assert abs(r2_1 - 1.0) < 0.0001, "R² devrait être 1.0 pour des prédictions parfaites"

# Test 2: Prédictions moyennes (R² = 0.0)
print("\n2️⃣  Test avec prédictions = moyenne:")
y_true_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y_pred_2 = np.array([0.3, 0.3, 0.3, 0.3, 0.3])  # Moyenne de y_true
r2_2 = r2_score(y_true_2, y_pred_2)
mae_2 = mean_absolute_error(y_true_2, y_pred_2)
print(f"   R² = {r2_2:.4f} (attendu: 0.0000)")
print(f"   MAE = {mae_2:.4f}")
assert abs(r2_2 - 0.0) < 0.0001, "R² devrait être 0.0 pour des prédictions constantes à la moyenne"

# Test 3: Bonnes prédictions (R² ~ 0.85)
print("\n3️⃣  Test avec bonnes prédictions:")
y_true_3 = np.array([0.10, 0.25, 0.35, 0.45, 0.60, 0.75, 0.85])
y_pred_3 = np.array([0.12, 0.23, 0.38, 0.42, 0.58, 0.78, 0.87])
r2_3 = r2_score(y_true_3, y_pred_3)
mae_3 = mean_absolute_error(y_true_3, y_pred_3)
print(f"   R² = {r2_3:.4f} (attendu: > 0.85)")
print(f"   MAE = {mae_3:.4f}")
assert r2_3 > 0.80, "R² devrait être > 0.80 pour de bonnes prédictions"

# Test 4: Mauvaises prédictions (R² < 0)
print("\n4️⃣  Test avec mauvaises prédictions:")
y_true_4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y_pred_4 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])  # Inversé
r2_4 = r2_score(y_true_4, y_pred_4)
mae_4 = mean_absolute_error(y_true_4, y_pred_4)
print(f"   R² = {r2_4:.4f} (attendu: < 0)")
print(f"   MAE = {mae_4:.4f}")
assert r2_4 < 0, "R² devrait être négatif pour des prédictions pires que la moyenne"

# Test 5: Vérification du formatage
print("\n5️⃣  Test du formatage d'affichage:")
r2_test = 0.8567
print(f"   R² avec .4f: {r2_test:.4f}")
print(f"   R² avec .2f: {r2_test:.2f}")
print(f"   R² avec .6f: {r2_test:.6f}")
print(f"   R² en %:     {r2_test*100:.2f}%")

# Test 6: Simulation données réelles (TO entre 0 et 1)
print("\n6️⃣  Test avec données simulées (TO entre 0 et 1):")
np.random.seed(42)
y_true_6 = np.random.uniform(0.3, 0.9, 100)  # TO entre 30% et 90%
noise = np.random.normal(0, 0.05, 100)  # Bruit gaussien
y_pred_6 = np.clip(y_true_6 + noise, 0, 1)  # Prédictions avec bruit
r2_6 = r2_score(y_true_6, y_pred_6)
mae_6 = mean_absolute_error(y_true_6, y_pred_6)
print(f"   R² = {r2_6:.4f}")
print(f"   MAE = {mae_6:.4f}")
print(f"   Plage y_true: [{y_true_6.min():.4f}, {y_true_6.max():.4f}]")
print(f"   Plage y_pred: [{y_pred_6.min():.4f}, {y_pred_6.max():.4f}]")

print("\n" + "=" * 80)
print("✅ TOUS LES TESTS SONT PASSÉS")
print("=" * 80)
print("\nSi le R² affiché dans l'entraînement est > 1 ou très anormal:")
print("1. Vérifiez que les données ne contiennent pas de NaN ou Inf")
print("2. Vérifiez qu'il n'y a pas de fuite de données (data leakage)")
print("3. Vérifiez que la normalisation est appliquée correctement")
print("4. Vérifiez que y_test et y_pred ont la même longueur")
print("=" * 80)

