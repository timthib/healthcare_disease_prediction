## Test d'un modèle de panel à effets fixes : le but est de comparer les évolutions intra pays : 
# On regarde si quand les caractéristiques d'un pays changent, par exemple s'il améliore son système de santé
# # est ce que le taux de mortalité infantile change aussi ?
# Mort_i,t​= b1* ​CP1_i,t ​+ b2* ​CP2_i,t​ + b3* ​CP3_i,t​ + α_i ​+ ϵ_i,t​

#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_panel_data(file_path):
    df = pd.read_csv(file_path)

    features = ['CP1_Index', 'CP2_Index', 'CP3_Index'] #Liste des colonnes explicatives 
    target = 'infant_mortality_rate' #Variable expliquée 

#Création d'un sous data frame avec seulement la variable Country, Year, CP1_Index, CP2_Index, CP3_Index et infant_mortality_rate
    df_panel = df[['Country', 'Year'] + features + [target]].dropna()

    return df_panel, features, target


def run_fixed_effects_regression(df, features, target):
    # Variables explicatives
    X = df[features]

    # Effets fixes pays
    country_dummies = pd.get_dummies(df['Country'], drop_first=True) #transforme la variable Country en variables binaires : rajout de colonnes avec ces variables
    # Chaque colonne indique si la ligne correspond à ce pays ou non
    # # on veut capturer l'effet spécifique à chaque pays : α_i 

    X = pd.concat([X, country_dummies], axis=1) # chaque ligne contient les variables explicatives + l’identifiant binaire du pays
    y = df[target] #variable à expliquer 

    model = LinearRegression() #les coeff associés aux indices PCA représentent l'effet intra pays (variation autour de la moyenne propre à chaque pays)
    model.fit(X, y)

    print("\n" + "="*70)
    print("--- MODÈLE DE PANEL À EFFETS FIXES (PAYS) ---")
    print("="*70)

    coeff_df = pd.DataFrame(
        model.coef_[:len(features)],
        index=features,
        columns=['Coefficient']
    )



    print("\nImpact des Indices (variation intra-pays) :")
    print(coeff_df)
    print("Intercept b0 :", model.intercept_)


    return model

if __name__ == "__main__":
    file_path_pca = 'healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv'

    df_panel, feats, targ = prepare_panel_data(file_path_pca)
    panel_model = run_fixed_effects_regression(df_panel, feats, targ)
