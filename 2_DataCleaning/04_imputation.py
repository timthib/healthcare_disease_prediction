import pandas as pd
import numpy as np

def impute_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # 1. Drop useless columns 
    # Based on our heatmap fig1, Gini and Poverty are mostly empty.
    cols_to_drop = ['gini_index', 'poverty_headcount']
    df.drop(cols_to_drop, axis=1, inplace=True)
    #axis=1 : delete columns (0 for rows), inplace = True to affect the original df (false : creates a copy to apply the drop)
    
    
    # 2. Imputation 
    # This is the trickiest part. We cannot just use the global average.
    # If we fill France's missing GDP with the world average, we ruin the data.
    # We must fill MISSING values based on THAT COUNTRY'S history.
    
    # Sort by Country and Year so time flows correctly (2010 -> 2011 -> 2012)
    df.sort_values(['Country', 'Year'], inplace=True)
    
    # Get list of numeric columns (we don't want to interpolate "Country" name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Year' in numeric_cols: 
        numeric_cols.remove('Year')
    
    def fix_country_data(group):
        """
        This function receives a DataFrame 'group' containing only one country's data.
        We fill the NaNs in this group.
        """
        #if [10,na,30] => [10,20,30]
        group[numeric_cols] = group[numeric_cols].interpolate(method='linear', limit_direction='both')
        #if fails because [10,na,na] for eg, => [10,10,10] ffill for gaps at the end, bfill for gaps at the begining
        group[numeric_cols] = group[numeric_cols].ffill().bfill()
        return group

    # Use groupby to apply the function to each country separately
    df_clean = df.groupby('Country', group_keys=False).apply(fix_country_data)
    
    # 3. Check
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"Remaining missing values after imputation: {remaining_missing}")
    
    if remaining_missing > 0:
        print("!!!!! Some countries might have NO data at all for certain columns.")
        drop_rows = df_clean[df_clean.isnull().any(axis=1)] #get columns where we have null values remaining
        print(drop_rows[['Country','Year']].groupby('Country').count())
        df_clean.dropna(inplace=True)
        #in that case drop rows

    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"Saved clean dataset")

if __name__ == "__main__":
    impute_data('healthcare_data_25countries/MASTER_DATASET.csv', 
                'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv')
