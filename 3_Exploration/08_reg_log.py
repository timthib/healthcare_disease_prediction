import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

# ----------------------------------------------------
# 1. Préparation des Données et de la Variable Cible
# ----------------------------------------------------
def prepare_data_for_modeling(file_path):
    # Charger le jeu de données qui contient les 3 composantes ACP
    df = pd.read_csv(file_path)
    
    # On travaille uniquement sur les données de la dernière année
    latest_year = df['Year'].max()
    # Nous utilisons dropna() pour retirer les pays pour lesquels il manque des données
    df_latest = df[df['Year'] == latest_year].copy().dropna().reset_index(drop=True) 
    
    # Définition de la variable cible binaire : Risque de Tuberculose
    # Nous utilisons le 75e percentile (Q3) de l'incidence TB comme seuil
    tb_incidence = df_latest['tb_incidence_per_100k']
    threshold_75 = tb_incidence.quantile(0.75)
    
    # Création de la variable cible (1 = Risque Élevé, 0 = Risque Faible)
    df_latest['High_TB_Risk'] = (tb_incidence > threshold_75).astype(int)
    
    print(f"Modèle basé sur les données de l'année : {latest_year}")
    print(f"Seuil de risque TB (Q3) établi à : {threshold_75:.2f} cas/100k.")
    print(f"Nombre de pays classés à Haut Risque (1) : {df_latest['High_TB_Risk'].sum()} / {len(df_latest)}")
    
    return df_latest

# ----------------------------------------------------
# 2. Modèle de Régression Logistique
# ----------------------------------------------------
def run_logistic_regression(df_model):
    
    # Variables Explicatives (X) :
    # Les trois indices ACP
    features = [
        'CP1_Index',  # Indice Richesse, Infrastructure et Santé Positive
        'CP2_Index',  # Indice Priorité des Dépenses vs. Infrastructure
        'CP3_Index'   # Indice Densité et Risque Démographique
    ]
    
    # Variable Cible (Y)
    target = 'High_TB_Risk'
    
    X = df_model[features]
    y = df_model[target]
    
    # Séparation des données en ensembles d'entraînement (Train) et de test (Test)
    # Stratify=y assure une répartition équilibrée des classes (0 et 1) dans les deux ensembles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Entraînement du modèle de Régression Logistique
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # Prédiction et Évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilité pour la classe 1 (Haut Risque)
    
    # Affichage des Résultats
    print("\n" + "="*70)
    print("--- RÉSULTATS DU MODÈLE DE RÉGRESSION LOGISTIQUE SUR L'ENSEMBLE DE TEST ---")
    print("="*70)
    
    print("\nCritères d'Évaluation (Classification Report) :")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nScore ROC AUC (Capacité de Discrimination) : {roc_auc_score(y_test, y_proba):.2f}")
    
    # Affichage de l'Importance des Variables (Coefficients)
    coefficients = pd.DataFrame(model.coef_[0], index=features, columns=['Coefficient'])
    
    # Calcul de l'Odd Ratio (e^Coefficient) pour une interprétation simple
    coefficients['Odd Ratio (e^Coefficient)'] = np.exp(coefficients['Coefficient'])
    
    print("\n--- Interprétation de l'Impact des Indices ACP (Odd Ratios) ---")
    print("Une augmentation d'une unité de l'indice :")
    print("- Augmente le risque TB si Odd Ratio > 1.")
    print("- Diminue le risque TB si Odd Ratio < 1.")
    print("-" * 50)
    print(coefficients.sort_values(by='Odd Ratio (e^Coefficient)', ascending=False))
    
    return model

if __name__ == "__main__":
    file_path_pca = 'healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv'
    
    # 1. Préparation des données
    df_ready = prepare_data_for_modeling(file_path_pca)
    
    # 2. Entraînement et évaluation du modèle
    logistic_model = run_logistic_regression(df_ready)