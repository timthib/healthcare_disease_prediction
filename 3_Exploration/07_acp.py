import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def perform_pca_three_components_final(file_path, n_components_to_keep=3):
    """
    Effectue l'ACP sur 12 variables (hors mortalité infantile), 
    affiche la composition des 3 premières CP et les ajoute au DataFrame.
    """
    df = pd.read_csv(file_path)
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy().reset_index(drop=True)
    
    # --- Sélection des 12 variables numériques (hors mortalité infantile) ---
    pca_cols = [
        'doctors_per_10k', 'hospital_beds_per_10k', 
        'maternal_mortality_rate', 'tb_incidence_per_100k', 'dtp3_vaccination_rate', 
        'nurses_per_10k', 'gdp_per_capita', 'pop_density', 'health_exp_usd', 
        'health_exp_gdp', 'population_total', 'urban_population'
    ]
    
    X = df_latest[pca_cols].values
    
    # 2. Standardisation des Données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Application de l'ACP
    pca = PCA(n_components=len(pca_cols))
    principal_components = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\n--- Analyse de la Variance Expliquée ---")
    print(f"Variance expliquée par les {n_components_to_keep} premières CP : {cumulative_variance[n_components_to_keep-1]:.2f}")
    
    # 4. Interprétation des Composantes Principales (Loadings)
    loadings = pd.DataFrame(pca.components_.T, 
                            columns=[f'CP{i}' for i in range(1, len(pca_cols) + 1)], 
                            index=pca_cols)
    
    # Impression explicite de la composition des 3 premières composantes
    print("\n" + "="*80)
    print(f"--- COMPOSITION EXPLICITE DES {n_components_to_keep} PREMIÈRES COMPOSANTES PRINCIPALES ---")
    print("= Les valeurs sont les poids (loadings) de chaque variable dans l'indice. =")
    print("="*80)
    print(loadings.iloc[:, :n_components_to_keep].to_markdown(floatfmt=".3f"))
    print("="*80 + "\n")
    
    # 5. Création et Ajout des N Indices au Jeu de Données
    new_cols = []
    for i in range(n_components_to_keep):
        col_name = f'CP{i+1}_Index'
        df_latest[col_name] = principal_components[:, i]
        new_cols.append(col_name)
    
    # Fusionner les nouveaux indices avec le DataFrame complet
    merge_cols = ['Country', 'Year'] + new_cols
    df = df.merge(df_latest[merge_cols], on=['Country', 'Year'], how='left')
    
    print(f"\nACP terminée. Les {n_components_to_keep} nouveaux indices ont été ajoutés au jeu de données.")
    return df

if __name__ == "__main__":
    file_path_clean = 'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv'
    file_path_output = 'healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv'
    
    master_df_with_index = perform_pca_three_components_final(file_path_clean)
    
    master_df_with_index.to_csv(file_path_output, index=False)
    print(f"Nouveau jeu de données avec 3 indices PCA sauvegardé sous {file_path_output}")