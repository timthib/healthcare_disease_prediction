import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# 1. Préparation des données (année 2024)
# ----------------------------------------------------
def prepare_2024_data(file_path):
    df = pd.read_csv(file_path)
    features = ['CP1_Index', 'CP2_Index', 'CP3_Index']
    target = 'infant_mortality_rate'
    
    # On ne garde que l'année 2024 et les données complètes
    df_2024 = df.copy()
    #df[df['Year'] == 2024].copy()
    df_2024 = df_2024.dropna(subset=features + [target])
    
    return df_2024, features, target

# ----------------------------------------------------
# 2. Entraînement d'un modèle par pays
# ----------------------------------------------------
def train_models_by_country(df, features, target):
    models = {}
    for country in df['Country'].unique():
        df_country = df[df['Country'] == country]
        X = df_country[features]
        y = df_country[target]
        model = LinearRegression().fit(X, y)
        models[country] = model
    return models

# ----------------------------------------------------
# 3. Fonction interactive pour prédiction + levier
# ----------------------------------------------------
def predict_mortality_and_levier(models, country, country_data, features):
    if country not in models:
        raise ValueError(f"Pas de modèle pour le pays {country}")
    
    model = models[country]
    X = np.array([[country_data[f] for f in features]])
    mortality_pred = model.predict(X)[0]
    
    # Levier le plus efficace pour réduire la mortalité
    coef = model.coef_
    levier_idx = np.argmin(coef)  # on cherche l'indice avec le coefficient le plus négatif
    levier = features[levier_idx]
    
    return mortality_pred, levier

# ----------------------------------------------------
# 4. Exemple d'utilisation
# ----------------------------------------------------
if __name__ == "__main__":
    file_path = 'healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv'
    
    # Préparer les données
    df_2024, features, target = prepare_2024_data(file_path)
    
    # Entraîner les modèles par pays
    models = train_models_by_country(df_2024, features, target)
    
    # Exemple : prédiction pour le pays "Japon"
    france_data = {
        'CP1_Index': 1.9,
        'CP2_Index': 1.3,
        'CP3_Index': 6.2
    }
    
    mortality_pred, levier = predict_mortality_and_levier(models, 'AFG', france_data, features)
    print(f"Mortalité prédite pour Japon : {mortality_pred:.2f}")
    print(f"Le levier le plus efficace pour réduire la mortalité : {levier}")

