import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def advanced_exploration(file_path):
    df = pd.read_csv(file_path)
    plt.figure(figsize=(14, 8))
    sns.lineplot(df, x='gdp_per_capita', y= 'infant_mortality_rate', hue='Country', sort=False)
    #hue = Country (different color per country)
    # sort = False (Connect points in order of Year, not by X value)
    
    # Also plot the scatter points on top so we see the years
    sns.scatterplot(data=df, x='gdp_per_capita', y='infant_mortality_rate', hue='Country', legend=False)
    
    # Label Start and End points
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country]
        
        # Start point
        start_row = country_data.sort_values('Year').iloc[0]
        plt.text(start_row['gdp_per_capita'], start_row['infant_mortality_rate'], 
                 str(start_row['Year']), fontsize=8, color='black')
                 
        # End point
        end_row = country_data.sort_values('Year').iloc[-1]
        plt.text(end_row['gdp_per_capita'], end_row['infant_mortality_rate'], 
                 str(end_row['Year']), fontsize=8, color='black', fontweight='bold')

    plt.yscale('log') # Log scale because GDP varies wildly
    plt.xscale('log')
    
    # Add readable ticks for the log scale
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    
    plt.title("Development Paths: The Race to Wealth & Health")
    plt.grid(True, which="both", ls="--")
    plt.savefig('Fig6_development_paths.png')

   


    """
    We want to see if there is a relationship between GDP and infant mortality rate to see if wealth is associated with health. 
    To do so, we fit a simple log linear regression: log(Mortality) ~ log(GDP), taking only the last year of data (no time series involved)
    Based on the data, we can have = log(moratility) = a + b*log(GDP) + epsilon, with epsilon the error term.
    For each country, we compute the error term (actual mortality rate - expected mortality rate) = performance score.
    A positive performance score means that the country is underperforming its GDP : on average and everything else being equal, the country is not efficient at using its wealth to improve health.
    a negative performance score means that the country is overperforming its GDP.
    """
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    plt.figure(figsize=(12, 8))
    
    # Fit a simple log linear regression: log(Mortality) ~ log(GDP)
    # We do this manually to get the expected mortality rate
    
    x = np.log(df_latest['gdp_per_capita'])
    y = np.log(df_latest['infant_mortality_rate'])
    
    # Polynomial fit (degree 1 = linear line)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    # Calculate Residuals (Actual - Expected)
    # Negative residual = Good (Mortality lower than expected, do more with less GDP) Positive residual = bad
    df_latest['expected_log_mortality'] = p(x)
    df_latest['performance_score'] = y - df_latest['expected_log_mortality']
    
   
    
    df_sorted = df_latest.sort_values('performance_score')
    sns.barplot(data=df_sorted, x='Country', y='performance_score', palette='coolwarm')
    
    
    plt.title(f"Healthcare Efficiency ({latest_year}): Who under/over-performs their GDP?")
    plt.ylabel("Performance Score (Lower is Better)")
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('Fig7_healthcare_efficiency.png')
    
    print("charts done")

# Fichier dynamics.py


def infant_mortality_analysis(file_path):
    df = pd.read_csv(file_path)
    latest_year = df['Year'].max()
    df_latest = df[df['Year'] == latest_year].copy()
    
    # ----------------------------------------------------
    # Plot 1: Vaccination vs. Infant Mortality (Prévention)
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line (using the last year's data)
    sns.regplot(data=df_latest, x='dtp3_vaccination_rate', y='infant_mortality_rate', 
                scatter_kws={'alpha':0.8}, line_kws={'color':'red', 'linestyle':'--'})
    
    plt.title(f"Couverture Vaccinale DTP3 vs. Mortalité Infantile ({latest_year})")
    plt.xlabel("Couverture Vaccinale DTP3 (%)")
    plt.ylabel("Taux de Mortalité Infantile (pour 1000 naissances)")
    plt.grid(True, axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Fig9_vaccine_vs_infant_mortality.png')
    
    # ----------------------------------------------------
    # Plot 2: Doctors vs. Infant Mortality (Infrastructure)
    # ----------------------------------------------------
    plt.figure(figsize=(12, 7))
    
    # Use log scale for Doctors and Mortality, and use GDP for color (hue)
    sns.scatterplot(data=df_latest, x='doctors_per_10k', y='infant_mortality_rate', 
                    hue='gdp_per_capita', size='gdp_per_capita', 
                    palette='viridis', sizes=(50, 500), alpha=0.7)
    
    # Add labels for a few extreme countries to provide context
    # Example to label the country with the highest and lowest infant mortality rate
    worst_performer = df_latest.loc[df_latest['infant_mortality_rate'].idxmax()]
    best_performer = df_latest.loc[df_latest['infant_mortality_rate'].idxmin()]
    
    plt.text(worst_performer['doctors_per_10k'], worst_performer['infant_mortality_rate'], 
             worst_performer['Country'], fontsize=9, ha='right', fontweight='bold')
    plt.text(best_performer['doctors_per_10k'], best_performer['infant_mortality_rate'], 
             best_performer['Country'], fontsize=9, ha='left', fontweight='bold')
    
    plt.xscale('log') 
    plt.yscale('log')
    plt.title(f"Accès aux Soins (Médecins) vs. Mortalité Infantile, Colorié par PIB ({latest_year})")
    plt.xlabel("Médecins pour 10k habitants (Échelle Log)")
    plt.ylabel("Taux de Mortalité Infantile (Échelle Log)")
    plt.legend(title='PIB par habitant')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('Fig10_doctors_vs_infant_mortality.png')
    
    print("Infant mortality analysis done (Fig9 and Fig10 generated).")


if __name__ == "__main__":
    file_path = 'healthcare_data_25countries/MASTER_DATASET_CLEAN.csv'
    advanced_exploration(file_path)
    #target_variable_analysis(file_path)
    infant_mortality_analysis(file_path) # NOUVELLE LIGNE AJOUTÉE
"""
if __name__ == "__main__":
    advanced_exploration('healthcare_data_25countries/MASTER_DATASET_CLEAN.csv')
"""
