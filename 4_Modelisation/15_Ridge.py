import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

# ----------------------------------------------------
# 1. Chargement du module précédent (Fichier 13)
# ----------------------------------------------------
def _load_lin_reg_module():
    module_path = Path(__file__).resolve().parent / "13_lin_reg_bis.py"
    spec = importlib.util.spec_from_file_location("lin_reg_bis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

lin_reg_mod = _load_lin_reg_module()
prepare_panel_data_with_lags = lin_reg_mod.prepare_panel_data_with_lags

# ----------------------------------------------------
# 2. Préparation des variables et effets fixes
# ----------------------------------------------------
def build_features_with_fe(panel_df):
    feature_cols = ['Year', 'PC1', 'PC2', 'PC3', 'Lag_Mortality']
    X_num = panel_df[feature_cols].copy()
    country_dummies = pd.get_dummies(panel_df['Country_ID'], prefix='Country', drop_first=True)
    X = pd.concat([X_num, country_dummies], axis=1)
    y = panel_df['Target_Mortality']
    return X, y

def walk_forward_split(panel_df, train_end_year=2020, test_start_year=2021):
    train_mask = panel_df['Year'] <= train_end_year
    test_mask = panel_df['Year'] >= test_start_year
    return train_mask, test_mask

# ----------------------------------------------------
# 3. Évaluation du Modèle Hybride Stable (Ridge)
# ----------------------------------------------------
def evaluate_hybrid_stable(panel_df):
    X, y = build_features_with_fe(panel_df)
    train_mask, test_mask = walk_forward_split(panel_df)

    X_train, X_test = X[train_mask].copy(), X[test_mask].copy()
    y_train, y_test = y[train_mask].copy(), y[test_mask].copy()

    # Seuil de spécialisation
    threshold = 30 
    
    # --- ENTRAINEMENT DES SPÉCIALISTES ---
    # Modèle Standard : Ridge avec régularisation légère
    mask_std = y_train <= threshold
    model_std = Ridge(alpha=1.0)
    model_std.fit(X_train[mask_std], y_train[mask_std])
    
    # Modèle Crise : Ridge avec régularisation FORTE pour éviter l'explosion
    mask_crisis = y_train > threshold
    model_crisis = Ridge(alpha=20.0) 
    model_crisis.fit(X_train[mask_crisis], y_train[mask_crisis])

    # --- PRÉDICTIONS ---
    # 1. Référence Linéaire Classique
    lin_base = LinearRegression()
    lin_base.fit(X_train, y_train)
    y_pred_lin = lin_base.predict(X_test)

    # 2. Modèle Hybride
    y_pred_hybrid = []
    for i in range(len(X_test)):
        if X_test.iloc[i]['Lag_Mortality'] > threshold:
            pred = model_crisis.predict(X_test.iloc[[i]])[0]
        else:
        
            pred = model_std.predict(X_test.iloc[[i]])[0]
        y_pred_hybrid.append(pred)
    y_pred_hybrid = np.array(y_pred_hybrid)

    # --- ÉVALUATION ---
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
    
    print("\n" + "="*50)
    print(f"RMSE Linéaire Classique : {rmse_lin:.4f}")
    print(f"RMSE Hybride Stable (Ridge) : {rmse_hybrid:.4f}")
    print("="*50)

    # --- VISUALISATION : RÉEL VS PRÉDIT ---
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_lin, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.title('Modèle Linéaire Classique (Base)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_hybrid, alpha=0.7, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    # On limite l'axe Y pour éviter de voir les résidus d'une explosion passée
    plt.ylim(0, y_test.max() + 10) 
    plt.title('Modèle Hybride Spécialisé (Stable)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('4_Modelisation/Fig24_hybrid_stable_comparison.png', dpi=300)

    # --- VISUALISATION : RÉSIDUS ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(y_test - y_pred_lin, kde=True, color='steelblue')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Résidus - Linéaire')

    plt.subplot(1, 2, 2)
    sns.histplot(y_test - y_pred_hybrid, kde=True, color='green')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Résidus - Hybride Stable')

    plt.tight_layout()
    plt.savefig('4_Modelisation/Fig25_hybrid_stable_residuals.png', dpi=300)
    
    return rmse_hybrid

if __name__ == "__main__":
    file_path_clean = 'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv'
    panel_df, _ = prepare_panel_data_with_lags(file_path_clean)
    evaluate_hybrid_stable(panel_df)