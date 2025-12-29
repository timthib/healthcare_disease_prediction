
# Plan de travail 

1. Trouver une problématique
2. Regrouper les données
3. Nettoyage et préparation des données
4. Explorer les données  
5. Statistiques descriptives 
6. Modélisation 
7. Rédaction du rapport 


1. Problématique 

La richesse d'un pays suffit-elle à garantir une faible mortalité infantile, ou l'organisation et l'intensité de l'offre de soins jouent-elles un rôle plus déterminant ?

Prédiction de la mortalité infantile et modélisation de ses déterminants selon les pays


2. Regrouper les données

- API de l'OMS : Données sur la prévalence des maladies, les taux de vaccination et les infrastructures de santé.

(https://www.who.int/data/gho/info/gho-odata-api)

- API de la Banque Mondiale : Données sur les indicateurs socio-économiques
(https://data.worldbank.org/topic/health) 

3. Nettoyage et préparation des données

- Nettoyage des données
- Gestion des valeurs manquantes
- Correction des types de variables
- Mise en forme et harmonisation des jeux de données

4. Exploration des données

- Inventaire de l’ensemble des variables disponibles
- Sélection des variables pertinentes
- Identification des variables exploitables
- Analyse des tendances temporelles et géographiques

5. Statistiques descriptives 
- Analyses bivariées et multivariées
- Étude des corrélations entre variables
- Analyse des évolutions conjointes
- Analyse en composantes principales (ACP) : Étant donné le grand nombre de variables fortement corrélées, une ACP est réalisée afin d’identifier trois composantes principales distinctes, selon le critère de Kaiser.


6. Modélisation 
- Régression linéaire du taux de mortalité infantile 
- Modèle prédictif du taux de mortalité infantile


## Ordre d'exécution et de lecture des notebooks  

1: Introduction
2: 1-DataCollection.ipynb
3: 2-DataCleaning.ipynb
4: 3-DataExploration.ipynb
5: 4-Modélisation
6: Conclusion