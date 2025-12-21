
# Plan 

1. Find problematic to adress
2. Gather data 
3. Exploration 
4. Preprocessing 
5. Analysis 
6. Modeling 
7. Report 


1. Problematic 
Map healthcare infrastructure gaps and predict disease outbreak vulnerability in underserved regions.

2. Gather data 
- [WHO Global Health Observatory API](https://www.who.int/data/gho/info/gho-odata-api) - Disease prevalence, vaccination rates, health infrastructure
- [World Bank Health API](https://data.worldbank.org/topic/health) - Healthcare expenditure, access metrics
- [OpenStreetMap API](https://wiki.openstreetmap.org/wiki/API) - Hospital and clinic locations


3. Data Cleaning 
- Clean data 
- Handle missing values 
- Set right types 
- Set right format 

4. Exploration 
- Get all columns 
- Filter relevant columns 
- Filter exploitable columns 
- See trends 
- Descriptive statistics 

5. Analysis 
- Bivariate and multivariate statistics 
- See correlations 
- See joint evolutions
- ACP because we have lots of variables that are quite correlated so we outline 3 (kaiser criteria) distinct characteristics. 
- See pair plot of ACP to see the nature of link between components and our interest variable : infant mortality rate 


6. Modelisation 
- Linear regression of infant morality rate 
- Model to predict infant mortality rate 
- get gradient to get the best lever to decrease it (combination)