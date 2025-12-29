import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def compute_pca_all_years(df, pca_cols, n_components=3):
    """
    On pool les données de toutes les années pour calculer les composantes principales.
    Cela permet de pouvoir comparer les composantes principales d'une année à l'autre.
    Si l'on avait calculé les composantes principales pour chaque année, on aurait obtenu des composantes principales différentes pour chaque année => difficle a interpréter.
    """
    df_result = df.copy()
    
    # Prepare pooled data (all years combined)
    # Fill missing values with year-specific means before pooling
    df_filled = df.copy()
    for col in pca_cols:
        df_filled[col] = df_filled.groupby('Year')[col].transform(lambda x: x.fillna(x.mean()))
    
    # Extract all available data for PCA fitting
    X_pooled = df_filled[pca_cols].dropna().values
    
    # Standardize pooled data
    scaler = StandardScaler()
    X_pooled_scaled = scaler.fit_transform(X_pooled)
    
    # Fit PCA on pooled data
    pca = PCA(n_components=n_components)
    pca.fit(X_pooled_scaled)
    
    print(f"\nACP variance expliquée (premieres {n_components} composantes):")
    for i in range(n_components):
        print(f"  CP{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
    print(f"  Cumulée: {pca.explained_variance_ratio_[:n_components].sum():.4f} ({pca.explained_variance_ratio_[:n_components].sum()*100:.2f}%)")
    
    # Visualisation des coefficients (loadings) et des contributions
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'CP{i+1}' for i in range(n_components)],
        index=pca_cols
    )
    
    # Barplot des loadings par composante
    plt.figure(figsize=(12, 8))
    for i in range(n_components):
        plt.subplot(n_components, 1, i + 1)
        sns.barplot(
            x=loadings.index,
            y=loadings.iloc[:, i],
            palette='viridis'
        )
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Poids')
        plt.title(f'Loadings de CP{i+1}')
    plt.tight_layout()
    plt.savefig('3_Exploration/Fig18_pca_loadings.png', dpi=300, bbox_inches='tight')
    
    # Barplot des contributions (variance expliquée)
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=[f'CP{i+1}' for i in range(n_components)],
        y=pca.explained_variance_ratio_[:n_components],
        palette='magma'
    )
    plt.ylabel('Part de variance')
    plt.title('Variance expliquée par composante')
    plt.tight_layout()
    plt.savefig('3_Exploration/Fig18_pca_variance.png', dpi=300, bbox_inches='tight')
    
    # Apply PCA to each year using the same transformation
    for year in df['Year'].unique():
        df_year = df_filled[df_filled['Year'] == year].copy()
        
        if len(df_year) == 0:
            continue
        
        # Extract features for this year
        X_year = df_year[pca_cols].values
        
        # Standardize using the same scaler fitted on pooled data
        X_year_scaled = scaler.transform(X_year)
        
        # Apply PCA transformation
        principal_components = pca.transform(X_year_scaled)
        
        # Store results
        for i in range(n_components):
            df_result.loc[df_result['Year'] == year, f'PC{i+1}'] = principal_components[:, i]
    
    return df_result

def prepare_panel_data_with_lags(file_path):
    """
    Create dataframe with:
    - Country_ID
    - Year
    - PC1, PC2, PC3
    - Lag_Mortality (infant_mortality_rate at t-1)
    - Target_Mortality (infant_mortality_rate at t)
    """
    df = pd.read_csv(file_path)
    
    # --- PCA Variables (12 variables excluding infant_mortality_rate) ---
    pca_cols = [
        'doctors_per_10k', 'hospital_beds_per_10k', 
        'maternal_mortality_rate', 'tb_incidence_per_100k', 'dtp3_vaccination_rate', 
        'nurses_per_10k', 'gdp_per_capita', 'pop_density', 'health_exp_usd', 
        'health_exp_gdp', 'population_total', 'urban_population'
    ]
    
    # Compute PCA for all years
    print("Computing PCA components for all years...")
    df = compute_pca_all_years(df, pca_cols, n_components=3)
    
    # Create Country_ID mapping
    unique_countries = df['Country'].unique()
    country_id_map = {country: idx + 1 for idx, country in enumerate(sorted(unique_countries))}
    df['Country_ID'] = df['Country'].map(country_id_map)
    
    # Sort by Country and Year to ensure proper lagging
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    # Create lagged mortality (t-1) and target mortality (t)
    df['Lag_Mortality'] = df.groupby('Country')['infant_mortality_rate'].shift(1)
    df['Target_Mortality'] = df['infant_mortality_rate']
    
    # Select only the required columns
    panel_df = df[['Country_ID', 'Year', 'PC1', 'PC2', 'PC3', 'Lag_Mortality', 'Target_Mortality']].copy()
    
    # Remove rows where Lag_Mortality is NaN (first year for each country)
    panel_df = panel_df.dropna(subset=['PC1', 'PC2', 'PC3', 'Lag_Mortality', 'Target_Mortality'])
    
    print(f"\n {len(panel_df)} observations")
    print(f"Countries: {len(panel_df['Country_ID'].unique())}")
    print(f"Years: {panel_df['Year'].min()} à {panel_df['Year'].max()}")
    
    return panel_df, country_id_map

def run_fixed_effects_regression(panel_df):
    """
    Model A: Fixed Effects Linear Regression
    Formula: Target_Mortality ~ Year + PC1 + PC2 + PC3 + EntityEffects
    Goal : see how well our principal components explain the target mortality
    """
    # Prepare features
    X_features = panel_df[['Year', 'PC1', 'PC2', 'PC3']].copy()
    
    # Add country fixed effects (EntityEffects)
    country_dummies = pd.get_dummies(panel_df['Country_ID'], prefix='Country', drop_first=True)
    
    # Combine features with fixed effects
    X = pd.concat([X_features, country_dummies], axis=1)
    y = panel_df['Target_Mortality']
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Evaluation metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Display results
    print("\nModel A: Fixed Effects Linear Regression")
    print(f"Formule: Target_Mortality ~ Year + PC1 + PC2 + PC3 + EntityEffects")
    print(f"\nR² (Coefficient de détermination): {r2:.4f}")
    print(f"RMSE (Erreur quadratique moyenne): {rmse:.4f}")
    
    # Coefficients for main features
    feature_names = ['Year', 'PC1', 'PC2', 'PC3']
    coeff_df = pd.DataFrame(
        model.coef_[:len(feature_names)],
        index=feature_names,
        columns=['Coefficient']
    )
    
    print("\nCoefficients principaux :")
    print(coeff_df)
    print(f"\nIntercept: {model.intercept_:.4f}")
    print(f"\nNombre d'effets fixes pays: {len(country_dummies.columns)}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2, label='Prédiction parfaite')
    plt.xlabel('Mortalité réelle')
    plt.ylabel('Mortalité prédite')
    plt.title('Modèle à effets fixes: Prédiction vs Réalité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus')
    plt.axvline(0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3_Exploration/Fig17_fixed_effects_regression.png', dpi=300, bbox_inches='tight')
    print("\nfig17 done")
    
    return model, coeff_df


def run_fixed_effects_regression_with_lag(panel_df):
    """
    Model B: Fixed Effects with lagged mortality
    Formula: Target_Mortality ~ Lag_Mortality + Year + PC1 + PC2 + PC3 + EntityEffects
    Goal: predict mortaility rate based on lag mortality rate + principal components
    """
    feature_cols = ['Year', 'PC1', 'PC2', 'PC3', 'Lag_Mortality']
    X_features = panel_df[feature_cols].copy()
    
    # Add country fixed effects
    country_dummies = pd.get_dummies(panel_df['Country_ID'], prefix='Country', drop_first=True)
    
    # Combine features
    X = pd.concat([X_features, country_dummies], axis=1)
    y = panel_df['Target_Mortality']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print("\nModel B: Fixed Effects with lagged mortality")
    print("Formula: Target_Mortality ~ Lag_Mortality + Year + PC1 + PC2 + PC3 + EntityEffects")
    print(f"\nR² (Coefficient de détermination): {r2:.4f}")
    print(f"RMSE (Erreur quadratique moyenne): {rmse:.4f}")
    
    # Coefficients for main features
    coeff_df = pd.DataFrame(
        model.coef_[:len(feature_cols)],
        index=feature_cols,
        columns=['Coefficient']
    )
    print("\nCoefficients principaux (avec lag) :")
    print(coeff_df)
    print(f"\nIntercept: {model.intercept_:.4f}")
    print(f"\nNombre d'effets fixes pays: {len(country_dummies.columns)}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y, y=y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', lw=2, label='Prédiction parfaite')
    plt.xlabel('Mortalité infantile réelle')
    plt.ylabel('Mortalité infantile prédite')
    plt.title('Modèle à effets fixes avec lag: Prédiction vs Réalité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y - y_pred
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    plt.title('Distribution des résidus')
    plt.axvline(0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3_Exploration/Fig19_fixed_effects_regression_with_lag.png', dpi=300, bbox_inches='tight')
    print("\nfig19 done (avec lag)")
    
    return model, coeff_df

if __name__ == "__main__":
    # Input file path
    file_path_clean = 'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv'
    
    # Prepare panel data with lags
    panel_df, country_map = prepare_panel_data_with_lags(file_path_clean)
    
    # Save the prepared panel dataframe
    output_path = 'healthcare_data_25countries/MASTER_DATASET_PANEL_LAGS.csv'
    panel_df.to_csv(output_path, index=False)
    print(f"\nPanel data done (avec lags): {output_path}")
    
    # Run fixed effects regression
    model_a, coefficients_a = run_fixed_effects_regression(panel_df)
    
    # Run fixed effects regression with lag
    model_b, coefficients_b = run_fixed_effects_regression_with_lag(panel_df)


