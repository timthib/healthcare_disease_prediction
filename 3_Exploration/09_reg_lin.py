import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------
# 1. Préparation des Données
# ----------------------------------------------------
def prepare_linear_data(file_path):
    df = pd.read_csv(file_path)
    latest_year = df['Year'].max()
    
    # On garde les colonnes nécessaires et on gère les valeurs manquantes
    features = ['CP1_Index', 'CP2_Index', 'CP3_Index']
    target = 'infant_mortality_rate'
    
    df_model = df[df['Year'] == latest_year].copy()
    df_model = df_model.dropna(subset=features + [target])
    
    return df_model, features, target

# ----------------------------------------------------
# 2. Modèle de Régression Linéaire
# ----------------------------------------------------
def run_linear_regression(df, features, target):
    X = df[features]
    y = df[target]
    
    # Division Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Évaluation
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n" + "="*70)
    print("--- RÉSULTATS DE LA RÉGRESSION LINÉAIRE (MORTALITÉ INFANTILE) ---")
    print("="*70)
    print(f"R² (Coefficient de détermination) : {r2:.4f}")
    print(f"RMSE (Erreur quadratique moyenne) : {rmse:.2f}")
    
    # Coefficients
    coeff_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
    print("\nImpact des Indices (Coefficients) :")
    print(coeff_df.sort_values(by='Coefficient'))

    # Affichage de l'intercept
    print("Intercept b0 :", model.intercept_)
    
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.xlabel('Valeurs Réelles (Mortalité Infantile)')
    plt.ylabel('Valeurs Prédites')
    plt.title('Prédiction vs Réalité : Mortalité Infantile')
    plt.savefig('Fig16_linear_regression_results.png')
    
    return model

if __name__ == "__main__":
    file_path_pca = 'healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv'
    
    df_lin, feats, targ = prepare_linear_data(file_path_pca)
    linear_model = run_linear_regression(df_lin, feats, targ)