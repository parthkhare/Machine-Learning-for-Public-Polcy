import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display


# Annexure I: Read Data
def import_data(filepath):
    '''
    Read csv files (current module is specified to csv files)
    Input: filepath
    Output: pandas Dataframe 
    '''

    return pd.read_csv(filepath)


# Annexure II A: Explore Features
def explore_features(df):
    '''
    Get broad overview of the dataset. Including data summary, likely outliers 
    and frequency distribution of all features.

    Inputs: pandas DataFrame(df)           
    Output: None
    '''

    # Check Data Types
    display('Feature Summary:',df.describe())
    
    # Compartive Chart
    print('Individual Feature Distribution I: Birds Eye')
    fig, ax = plt.subplots(figsize=(11,9)) 
    sns.boxplot(data=df, orient="h", palette="Set2")
    
    return None


# Annexure II B: Explore Feature Interactions
def explore_finteractions(df, cont_var):
    '''
    Generates feature interactions via correlation heatmap and pairwise plots 
    Inputs: pandas DataFrame (df)
            continous attributes in the dataset (cont_var)
    Output: None
    '''

    # Pairwise Interactions I: Overall
    print('Pairwise Feature Distribution: Variable Histograms')
    fig, ax = plt.subplots(figsize=(11,9)) 
    corr = df.corr()
    sns.heatmap(corr,annot=True,cmap="YlGnBu")
    
    # Pairwise Interactions II: Identified Continous Features 'kde/reg'
    print('Pairwise Feature Distribution: Pairplot')
    sns.pairplot(df, vars = cont_var,  diag_kind = 'kde', 
                 plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                 size = 4)

    return None


# Annexure III: Preprocess Data Missing Values
def treat_missing(df, mean=True):
    '''
    Shows missing value frequency in data, fill NA values feature mean/median
    Input: pandas DataFrame(df)
    Output: DataFrame without missing values(df_nm)
    '''

    # Summary of missing Values in the dataset
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print('Missing data summary')
    display(missing_data.head(20))
    
    # Fill missing with mean/median value
    if mean:
        df_nm = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    else:
        df_nm = df.apply(lambda x: x.fillna(x.median()),axis=0)
    return df_nm


# Annexure IV.A: Transform Data: Continous to Discrete
def cont_discr(df, var_tr, bins=5):
    '''
    Discretize a continuous variable in the data set based on user defined bins
    Inputs: pandas DataFrame(df)
            features to examine interactions of(cont_var)
            number of bins(bins, default=5)
    Output: Transformed DataFrame with converted discrete features(df_tr)
    '''
    # Create copy of the data to augment variables
    df_tr = df.copy(deep=True)

    for feature in var_tr:
        df_tr[feature] = pd.cut(df[feature], bins, labels=list(range(bins)))
        df_tr[feature] = df_tr[feature].astype('str')  

    return df_tr 


# Annexure IV.B: Transform Data: Cat to Dummy
def cat_dummy(df, cols_to_transform):
    '''
    Create dummy variables
    Inputs: pandas DataFrame(df)
            list of features to transform(cols_to_transform)
            number of bins(bins, default=5)
            type of split/cut(quantile if True)
    Output: Transformed DataFrame with converted dummy features(df_td)
    '''
    # Create copy of the data to augment variables
    df_td = df.copy(deep=True)
    df_td = pd.get_dummies(df, dummy_na=True, columns = cols_to_transform)

    return df_td