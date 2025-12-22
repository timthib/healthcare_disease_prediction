import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


def _load_lin_reg_module():
    """
    Load the 13_lin_reg_bis module despite its numeric filename.
    """
    module_path = Path(__file__).resolve().parent / "13_lin_reg_bis.py"
    spec = importlib.util.spec_from_file_location("lin_reg_bis", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


lin_reg_mod = _load_lin_reg_module()
prepare_panel_data_with_lags = lin_reg_mod.prepare_panel_data_with_lags


def build_features_with_fe(panel_df):
    """
    Construct feature matrix with:
    - Year
    - PC1, PC2, PC3
    - Lag_Mortality
    - Country fixed effects (one-hot, drop_first to avoid collinearity)
    """
    feature_cols = ['Year', 'PC1', 'PC2', 'PC3', 'Lag_Mortality']
    X_num = panel_df[feature_cols].copy()
    country_dummies = pd.get_dummies(panel_df['Country_ID'], prefix='Country', drop_first=True)
    X = pd.concat([X_num, country_dummies], axis=1)
    y = panel_df['Target_Mortality']
    return X, y


def walk_forward_split(panel_df, train_end_year=2020, test_start_year=2021):
    """
    Time-based split:
    - Train: years <= train_end_year
    - Test : years >= test_start_year
    """
    train_mask = panel_df['Year'] <= train_end_year
    test_mask = panel_df['Year'] >= test_start_year

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Train or test split is empty. Check year ranges in the dataset.")

    return train_mask, test_mask


def evaluate_models_walk_forward(panel_df):
    """
    Compare two models on a walk-forward split:
    - Linear regression with lag + country fixed effects
    - XGBoost regressor with the same inputs
    Returns a dict with RMSE metrics and predictions, and saves comparison plots.
    """
    X, y = build_features_with_fe(panel_df)
    train_mask, test_mask = walk_forward_split(panel_df, train_end_year=2020, test_start_year=2021)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # -------- Model 1: Linear regression (lag + FE) --------
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))

    # -------- Model 2: XGBoost regressor --------
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='rmse'
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

    print("\nÉvaluation walk-forward (Train: <=2020, Test: 2021-2024)")
    print(f"RMSE - Linear FE + Lag : {rmse_lin:.4f}")
    print(f"RMSE - XGBoost        : {rmse_xgb:.4f}")

    # -------- Visualizations for explainability --------
    # 1) Predicted vs Actual for both models
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_lin, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Prédiction parfaite')
    plt.xlabel('Mortalité réelle (test)')
    plt.ylabel('Mortalité prédite (linéaire)')
    plt.title('Linéaire FE + Lag: Prédiction vs Réalité')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_xgb, alpha=0.7, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2, label='Prédiction parfaite')
    plt.xlabel('Mortalité réelle (test)')
    plt.ylabel('Mortalité prédite (XGBoost)')
    plt.title('XGBoost: Prédiction vs Réalité')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('4_Modelisation/Fig20_pred_vs_actual_lin_xgb.png', dpi=300, bbox_inches='tight')

    # 2) Residuals distribution for both models
    residuals_lin = y_test - y_pred_lin
    residuals_xgb = y_test - y_pred_xgb

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals_lin, kde=True, bins=25, color='steelblue')
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Résidus (réel - prédit)')
    plt.ylabel('Fréquence')
    plt.title('Résidus - Linéaire FE + Lag')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    sns.histplot(residuals_xgb, kde=True, bins=25, color='orange')
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Résidus (réel - prédit)')
    plt.ylabel('Fréquence')
    plt.title('Résidus - XGBoost')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('4_Modelisation/Fig21_residuals_lin_xgb.png', dpi=300, bbox_inches='tight')

    return {
        'rmse_linear_fe_lag': rmse_lin,
        'rmse_xgboost': rmse_xgb,
        'y_test': y_test,
        'y_pred_lin': y_pred_lin,
        'y_pred_xgb': y_pred_xgb,
    }


if __name__ == "__main__":
    file_path_clean = 'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv'

    # Prepare panel data (includes PCs, lag mortality, and country IDs)
    panel_df, _ = prepare_panel_data_with_lags(file_path_clean)

    # Evaluate models on walk-forward split (2010-2020 train, 2021-2024 test)
    evaluate_models_walk_forward(panel_df)

