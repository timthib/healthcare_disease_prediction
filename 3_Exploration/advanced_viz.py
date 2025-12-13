import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def further_visualizations(file_path):
    df = pd.read_csv(file_path)
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    # ----------------------------------------------------
    # Plot 11: Urbanisation vs. TB Incidence (Densité et Maladie)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Visualiser la relation entre la population urbaine (en %) et l'incidence de la TB
    sns.regplot(data=df_latest, x='urban_population', y='tb_incidence_per_100k', 
                scatter_kws={'alpha':0.8}, line_kws={'color':'green', 'linestyle':'-'})
    
    plt.title(f"Urbanisation vs. Incidence de la Tuberculose ({latest_year})")
    plt.xlabel("Pourcentage de Population Urbaine")
    plt.ylabel("Incidence de la TB pour 100k habitants")
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig11_urban_vs_tb_incidence.png')
    
    print("Urbanization vs TB incidence chart generated (Fig11).")

    # ----------------------------------------------------
    # Plot 12: Boxplots - Richesse (GDP Quartiles) vs. Santé (Mortalité Infantile)
    # ----------------------------------------------------
    
    # Créer des catégories de richesse (Quartiles de PIB)
    df_latest['GDP_Quartile'] = pd.qcut(df_latest['gdp_per_capita'], q=4, labels=['Q1 (Pauvre)', 'Q2', 'Q3', 'Q4 (Riche)'])
    
    plt.figure(figsize=(10, 7))
    
    # Boxplot de la mortalité infantile par quartile de PIB
    sns.boxplot(data=df_latest, x='GDP_Quartile', y='infant_mortality_rate', 
                palette='YlGnBu')
    
    plt.title(f"Distribution de la Mortalité Infantile par Quartile de Richesse ({latest_year})")
    plt.xlabel("Quartile de PIB par habitant")
    plt.ylabel("Taux de Mortalité Infantile (pour 1000 naissances)")
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig12_gdp_quartile_vs_mortality_boxplot.png')
    
    print("GDP Quartile vs Infant Mortality boxplot generated (Fig12).")


if __name__ == "__main__":
    # Assurez-vous d'appeler cette nouvelle fonction à la fin de votre script dynamics.py
    further_visualizations('healthcare_data_25countries/MASTER_DATASET_CLEAN.csv')