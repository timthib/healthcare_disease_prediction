import pandas as pd
import os
import glob


def load_and_clean_who(filepath, indicator):
    """
    Reads a WHO dataset and cleans it to just 3 columns: Country, Year, Value
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns={'SpatialDim': 'Country', 'TimeDim': 'Year', 'NumericValue' : indicator})
    df = df[['Country', 'Year', indicator]]
    
    # Handle duplicates (sometimes WHO has multiple entries per year due to gender split in 'Dim1')
    # For now,we just take the mean if there are duplicates for Country/Year
    df = df.groupby(['Country', 'Year'], as_index=False)[[indicator]].mean()
        
    return df

def load_and_clean_wb(filepath, indicator):
    """
    Reads a World Bank dataset and cleans it.
    """
    df = pd.read_csv(filepath)
    
    # we drop 'country' column to keep 'economy' that corresponds to the "WHO" country column  
    if 'Country' in df.columns:
        df = df.drop(columns=['Country'])

    df.rename(columns={'economy': 'Country', 'Time': 'Year', df.columns[-1]: indicator}, inplace=True)
    df = df[['Country', 'Year', indicator]]
    return df

def main():
    # 1. Map file names to indicators column names
    who_files = {
        'doctors': 'doctors_per_10k',
        'hospital_beds': 'hospital_beds_per_10k',
        'infant_mortality': 'infant_mortality_rate',
        'maternal_mortality': 'maternal_mortality_rate',
        'tb_incidence': 'tb_incidence_per_100k',
        'dtp3_vaccination': 'dtp3_vaccination_rate',
        'nurses': 'nurses_per_10k',
    }
    
    wb_files = {
        'gdp_per_capita': 'gdp_per_capita',
        'population_density': 'pop_density',
        'health_exp_capita': 'health_exp_usd',
        'health_exp_gdp': 'health_exp_gdp',
        'population_total': 'population_total',
        'urban_population': 'urban_population',
        'poverty_headcount': 'poverty_headcount',
        'gini_index': 'gini_index',
    }

    master_df = pd.DataFrame(columns=['Country', 'Year'])

    # 2. Process WHO Data
    print("--- Processing WHO Data ---")
    base_path_who = 'healthcare_data_25countries/who'
    
    df_to_merge = []
    
    for filename, col_name in who_files.items():
        path = os.path.join(base_path_who, f"{filename}.csv")
        if os.path.exists(path):
            clean_df = load_and_clean_who(path, col_name)
            df_to_merge.append(clean_df)
        else:
            print(f"WARNING: File not found {path}")

    # 3. Process World Bank Data
    base_path_wb = 'healthcare_data_25countries/worldbank'
    
    for filename, col_name in wb_files.items():
        path = os.path.join(base_path_wb, f"{filename}.csv")
        if os.path.exists(path):
            clean_df = load_and_clean_wb(path, col_name)
            df_to_merge.append(clean_df)
    
    if df_to_merge:
        master_df = df_to_merge[0]
        for i in range(1, len(df_to_merge)):
            master_df = pd.merge(master_df, df_to_merge[i], on=['Country','Year'],how='outer') #we need outer join in case some countries miss data for some years
            pass 

    print(master_df.head())
    
    master_df.to_csv('healthcare_data_25countries/MASTER_DATASET.csv', index=False)
    print("master dataset saved to healthcare_data_25countries/MASTER_DATASET.csv")

if __name__ == "__main__":
    main()

