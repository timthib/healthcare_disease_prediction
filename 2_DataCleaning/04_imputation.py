import pandas as pd
import numpy as np

def impute_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # 1. Suppression des colonnes inutiles 
    # D'après notre heatmap fig1, Gini et Poverty sont majoritairement vides.
    cols_to_drop = ['gini_index', 'poverty_headcount']
    df.drop(cols_to_drop, axis=1, inplace=True)
    # axis=1 : supprime les colonnes (0 pour les lignes), inplace=True pour modifier le df original 

    
    # Trier par pays et par année 
    df.sort_values(['Country', 'Year'], inplace=True)
    
    # Obtenir la liste des colonnes numériques (on ne veut pas interpoler le nom du "Country")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Year' in numeric_cols: 
        numeric_cols.remove('Year')
    
    def fix_country_data(group):
        """
        Cette fonction reçoit un DataFrame 'group' contenant les données d'un seul pays.
        On remplit les NaN dans ce groupe.
        """
        # si [10,na,30] => [10,20,30]
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear', limit_direction='both')
        # si ça échoue parce que [10,na,na] par exemple => [10,10,10] ffill pour combler les lacunes à la fin, bfill pour combler les lacunes au début
        group[numeric_cols] = group[numeric_cols].ffill().bfill()
        return group

    # Utiliser groupby pour appliquer la fonction à chaque pays séparément
    df_clean = df.groupby('Country', group_keys=False).apply(fix_country_data)
    
    # 3. Vérification
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"Valeurs manquantes restantes après imputation: {remaining_missing}")
    
    if remaining_missing > 0:
        print("!!!!! Certains pays pourraient ne pas avoir de données du tout pour certaines colonnes.")
        drop_rows = df_clean[df_clean.isnull().any(axis=1)] # récupérer les lignes avec des valeurs nulles restantes
        print(drop_rows[['Country','Year']].groupby('Country').count())
        df_clean.dropna(inplace=True)
        # dans ce cas, supprimer les lignes

    
    # Sauvegarder 
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset nettoyé sauvegardé")

if __name__ == "__main__":
    impute_data('healthcare_data_25countries/MASTER_DATASET.csv', 
                'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv')
