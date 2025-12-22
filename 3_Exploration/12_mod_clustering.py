## Modèle de clustering à partir des données de l'ACP, utilisation de la méthode des K means

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le dataset avec les indices ACP
df = pd.read_csv('healthcare_data_25countries/MASTER_DATASET_PCA_3CP_ALL_VARS.csv')

# On travaille uniquement sur une année (ex. 2024)
df_2024 = df[df['Year'] == 2024].copy().dropna(subset=['CP1_Index','CP2_Index','CP3_Index'])

# Variables pour le clustering
X = df_2024[['CP1_Index','CP2_Index','CP3_Index']].values

# Standardiser les données pour que toutes les composantes aient le même poids
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.cluster import KMeans

# Définir le nombre de clusters (à ajuster selon le contexte)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_2024['Cluster'] = kmeans.fit_predict(X_scaled)


## AFFICHAGE FIGURE ###

# Afficher les pays et leur cluster
print(df_2024[['Country','CP1_Index','CP2_Index','CP3_Index','Cluster']])


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df_2024['CP1_Index'], df_2024['CP2_Index'], df_2024['CP3_Index'],
    c=df_2024['Cluster'], cmap='Set1', s=100
)
ax.set_xlabel('CP1_Index')
ax.set_ylabel('CP2_Index')
ax.set_zlabel('CP3_Index')
plt.title('Clustering des pays selon les indices ACP')
plt.show()




import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Charger le dataset
df = pd.read_csv('MASTER_DATASET_PCA_3CP_ALL_VARS.csv')
df_2024 = df[df['Year']==2024].dropna(subset=['CP1_Index','CP2_Index','CP3_Index'])

# Clustering
X = df_2024[['CP1_Index','CP2_Index','CP3_Index']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df_2024['Cluster'] = kmeans.fit_predict(X_scaled)

# Titre
st.title("Analyse des pays selon les indices ACP")

# Choisir un pays
pays = st.selectbox("Choisir un pays", df_2024['Country'].unique())
pays_data = df_2024[df_2024['Country']==pays].iloc[0]

# Affichage cluster
st.write(f"Le pays {pays} appartient au cluster {pays_data['Cluster']}")

# Affichage des leviers prioritaires
indices = ['CP1_Index','CP2_Index','CP3_Index']
leviers = pays_data[indices].sort_values()  # par exemple les indices les plus faibles
st.write("Indices ACP du pays :")
st.dataframe(leviers)
st.write("Levier prioritaire (indice le plus faible) :", leviers.idxmin())

# Optionnel : modifier les indices pour simuler l'effet
st.subheader("Simulation de scénario")
for idx in indices:
    pays_data[idx] = st.slider(f"{idx}", min_value=float(df_2024[idx].min()), max_value=float(df_2024[idx].max()), value=float(pays_data[idx]))

st.write("Indices modifiés :", pays_data[indices])
