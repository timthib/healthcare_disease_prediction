"""
This files allow us to visualize missing values across countires and columns of interest 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_missing_values(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Missing values check
    print("\n--- Missing Values Summary ---")
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    
    summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_percent
    })
    print(summary.sort_values('Missing %', ascending=False))
    
    # 2. Visualization: The Heatmap
    #To see patterns in missing data, like are we missing entire years or countries or just random cells ?

    #Fig 1 : see number of missing values per country per feature
    plt.figure(figsize=(12,8))
    plt.title("Fig1 : Missing Data Heatmap (Clear=Missing, Dark=Present)")
    df1 = df.set_index('Country')
    country_missing2 = df1.groupby('Country').apply(lambda x: x.isnull().sum() / x.shape[0]*x.shape[1] *100)
    sns.heatmap(country_missing2,  cbar=False, yticklabels=False, cmap='viridis')
    plt.tight_layout()
    plt.savefig('Fig1_missing_values_per_col_per_country.png')
    
    """
    result : gini index and poverty head_count are to get rid of, to many missing values accross all countries
    #tb_incidence_per_100k appears to be unavailable for some countries
    #doctors_per_10k and maternity_mortality_rate are missing quite often but might still be usefull 
    """

    #Fig 2 : see missing values across (country, year) pair
    """
    if different pattent than fig1, it means that we miss some years. 
    Result : same pattern, since health system are at the country level, a country might face difficulties
    gathering data for a given year but this would be independant from other countrie's ability to gather data.
    """
    plt.figure(figsize=(12, 8))    
    #we set country and year as index so that we don't check null values for these columns but only for the indicators 
    df_heat = df.set_index(['Country', 'Year']).isnull()
    # cbar=False cleans it up
    sns.heatmap(df_heat, cbar=False, yticklabels=False, cmap='viridis')
    plt.title("Missing Data Heatmap (Yellow=Missing, Purple=Present)")
    plt.tight_layout()
    plt.savefig('Fig2_missing_values_heatmap.png')

    
    # Fig3 :  What countries have the most missing values 
    """
    The goal is to see what countries might be to remove from the data. 
    Arbitraly, we decide that if we have more than 20% missing values for a country, we don't use the country for the analysis.
    In the previous fig1, the fact that some rows are very clear across multiple columns signals that some countries migh be to set aside from the data.
    """
    plt.figure(figsize=(10, 6))
    # Calculate % missing per country
    """
    Take the df,  create mini df for each countries, count all missing values within each mini df,
    x.isnull().sum() gives the count of null values per column, add a .sum() to get the total number of null values 
    we divide by number of cells to get the % of missing values 
     We sum all missing values for a country, divided by total cells for that country
    """
    country_missing = df.groupby('Country').apply(lambda x: x.isnull().sum().sum() / (x.shape[0] * x.shape[1]) * 100)
    country_missing = country_missing.sort_values(ascending=False)
    
    sns.barplot(x=country_missing.index, y=country_missing.values, palette='Reds_r')
    plt.xticks(rotation=45)
    plt.ylabel("Total % Missing Data")
    plt.title("Data Quality by Country (Higher is Worse)")
    plt.tight_layout()
    plt.savefig('Fig3_missing_values_by_country.png')

    print("Figures done")


if __name__ == "__main__":

    analyze_missing_values('healthcare_data_25countries/MASTER_DATASET.csv')

