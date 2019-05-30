import pandas as pd
import seaborn as sns
import warnings
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import display


# Annexure I A: Read Data
def import_data(filepath):
    '''
    Read csv files (current module is specified to csv files)
    Input: filepath
    Output: pandas Dataframe 
    '''

    return pd.read_csv(filepath)

# Annexure I B: Convert multiple column datattypes
def trnf_dtype(df, var_list, to_type='cat'):
    '''
    Convert multiple variable to specified datatype
    Input: pandas Dataframe, variable specification
    Output: pandas Dataframe 
    '''
    if to_type == 'cat':
        for col in var_list:
            df[col] = df[col].astype('category')
    elif to_type == 'cont':
        for col in var_list:
            df[col] = df[col].astype('float32')
    elif to_type == 'date':
        for col in var_list:
            df[col] = pd.to_datetime(df[col])

    return df


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
    df_cont = df[cont_var] 
    fig, ax = plt.subplots(figsize=(11,9)) 
    corr = df_cont.corr()
    sns.heatmap(corr,annot=True,cmap="YlGnBu")
    
    # Pairwise Interactions II: Identified Continous Features 'kde/reg'
    print('Pairwise Feature Distribution: Pairplot')
    sns.pairplot(df_cont, vars = cont_var,  diag_kind = 'kde', 
                 plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                 size = 4)

    return None

# Annexure II C: Explore Correlations
def plot_correlations(df, x, y, hue = None, fitreg = False):
    '''
    Make a scatter plot of using the Seaborn lmplot function
    Inputs: pandas DataFrame (df) 
            x, y (strs): Variables to identify as x and y
           hue (str): Optional third variable to color datapoints
           fitreg (bool): Option to include a fitline
    No return, shows a scatter plot
    '''
    sns.lmplot(x, y, hue = hue, fit_reg = fitreg, data = df)
    plt.title(x + ' vs ' + y)
    plt.show()

# Annexure III A: Preprocess Data Missing Values
def treat_missing(df, col, mean=True):
    '''
    Shows missing value frequency in data, fill NA values feature mean/median
    Input: pandas DataFrame(df)
           mean/median (default=mean)
           col=specify (default=all columns)
    Output: DataFrame without missing values(df)
    '''

    # Summary of missing Values in the dataset
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    #print('Missing data summary')
    display(missing_data.head(10))
    
    # Find columns with missing values
    # missing = df.isnull().any()
    # missing_col = list(isnull[isnull == True].index)

    # Fill missing with mean/median value 
    if mean:
        for var in col:
            # df_nm = df.apply(lambda x: x.fillna(x.mean()),axis=0)
            # df = trnf_dtype(df, var_list=col, to_type='cat')
            col_mean = df[var].mean()
            df[var].fillna(col_mean, inplace = True)
    else:
        df = trnf_dtype(df, var_list=col, to_type='cat') # ensure if not transformed yet
        # df = df.apply(lambda x: x.fillna(x.median()),axis=0)
        for var in col:
            print('varible treated:', var)
            col_median = df[var].median()
            df[var].fillna(col_median, inplace = True)

    return df

# Annexure III A: Preprocess Data Missing Values
def scale_features(df, col):
    '''
    Normalise features, if they are skewed
    Input: pandas DataFrame(df)
    Output: DataFrame with normalised features(df)
    '''

# Annexure III C: 
def get_zscore(df, col, zparam=1.96):
    '''
    Calculate z-score for data and then store as outliers for those with 
    more than a given z-score from mean
    
    Inputs:
        df pandas DataFrame(df)
        col (str) the column name
        zparam (float) the z-score to consider an outlier. Defaults to 1.96 (p=0.05)
        
    Output:
        df pandas DataFrame(df)
        zscr (str) name of the new column
        outliers (list) list of indices pertaining to outlier set
    '''
    zscr = str(col) + "_zscore"
    currz = df[col].apply(lambda x: (x - df[col].mean()) / df[col].std())
    outliers = df.index[abs(currz) >= zparam ].tolist()

    return zscr, outliers



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