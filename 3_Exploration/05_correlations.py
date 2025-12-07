import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_correlations(file_path):
    # Load Clean Data
    df = pd.read_csv(file_path)
    numeric_df = df.drop(columns=['Country', 'Year']) #we drop non numeric and year because correlation with year is trivial (trend across time)
    corr_matrix = numeric_df.corr() 
    plt.figure(figsize=(12, 10))
    
    # mask to only see the coefficient once, symetric matrix and diagonal elements = 1
    mask = np.triu(np.ones_like(corr_matrix,dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True,fmt=".2f", cmap='coolwarm',center=0)
    #annot = true to show numbers, format with 2 decimals, center=0 so that no correlation is neutral, cmap= coolwars so taht red =1, blue = -1
    plt.title("Correlation Matrix: What drives Health?")
    plt.tight_layout()
    plt.savefig('Fig4_correlation_matrix.png')
    
    
   
   #sns.pairplot(df, kind='reg') #reg adds a regression line
   #plt.savefig('Fig_pairplot_all_columns.png')
    """
    From this pairplot, we decide to focus on certain columns of interest that present an interest :
    """
    cols_of_interest = ['gdp_per_capita', 'health_exp_usd', 'doctors_per_10k', 'infant_mortality_rate', 'maternal_mortality_rate','tb_incidence_per_100k', 'dtp3_vaccination_rate']
    sns.pairplot(df[cols_of_interest], kind='reg')
    plt.savefig('Fig5_pairplot_interest_columns.png')
    
    print("correlation and pairplot done")

if __name__ == "__main__":
    analyze_correlations('healthcare_data_25countries/MASTER_DATASET_CLEAN.csv')

