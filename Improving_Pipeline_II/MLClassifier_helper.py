from __future__ import division
from IPython.display import display

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import optimize
import time
import seaborn as sns

from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize, scale
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import MLPreprocess_helper as mh


# --- Codes on Standard Model Fitting adapted/modified from Rayid Ghani's magicloops repository (https://github.com/rayidghani/magicloops) ---- #

# ------------------------------------------------
# # Section I: Restructing Data & Metrics
# ------------------------------------------------

# Training Test Data
# ------------------------------------------------
def feature_eng(df, missing_var, dum_var, bin_var, norm_var, maj_cities):
    '''
    Feature engineering on training data, allowing dynamically/sequential implementation of temporal test and train splits 
    Input: Training dataset
    Output: splitted test and train data based start and end time (pandas Dataframe)
    '''
    print('\n Feature engineering module')
    # Treat Missing Values
    print('Treating Missing Values')
    df = mh.treat_missing(df, col=missing_var)

    # Generate features
    print('Major cities dummy')
    df['maj_cities'] = np.where(df['school_city'].isin(maj_cities), 1, 0)
    print('Gender dummy')
    print(df.groupby(['teacher_prefix']).size())
    df['gender'] = np.where(df['teacher_prefix'].isin(['Mrs.', 'Ms.']), 1, 0)

    # Create Dummy Variables
    print('Creating Dummy labels for categoricals')
    df = mh.cat_dummy(df, dum_var)

    # Binary Variable
    print('Creating Binary labels')
    for var in bin_var:
        df[var] = np.where(df[var] == 't', 1, 0)

    # Normalise
    print('Normalise Skewed Variables \n')
    for var in norm_var:
        df[var] = scale(df[var])

    return df

def temporal_train_test_split(df, train_start, train_end, test_start, test_end, time_var, pred_var):
    '''
    Split the data based on temporal windows used in clf_wrapper to integrate with clf loops
    Input: Dataset to be split, starting and end period
    Output: splitted test and train data based start and end time (pandas Dataframe)
    '''
    train_df = df[(df[time_var] >= train_start) & (df[time_var] <= train_end)]
    y_train = train_df[pred_var]
    X_train = train_df.drop([pred_var, time_var], axis = 1)
    test_df = df[(df[time_var] >= test_start) & (df[time_var] <= test_end)]
    y_test = test_df[pred_var]
    X_test = test_df.drop([pred_var, time_var], axis = 1)

    return X_train, y_train, X_test, y_test




# ML Validation Metrics
# ------------------------------------------------

def joint_sort_descending(l1, l2):
    '''
    Sorts the predicted values
    Input: unsorted vectors
    Output: vector sorted in descending order
    '''

    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Convert predicted probability values to binary 0,1 used for Confusion matrix/F1
    Method is complaint with using percentile thresholds instead in terms of making them representativewrt to sample size
    Note this metric is useful only for larger sizes of data, with smaller size values len(y_scores) might bias cutoff
    Input: Column (Pandas DF/Series) Threshold
    Output: Transformed Binary Outcome  
    '''

    cutoff_index = int(len(y_scores) * (k / 100.0))   # Multiply for feasible class representation {not recommended for small size}
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    '''
    Find precison at given threhold 
    Input: Binary outcome from test data & Predicted probabilities from model on training data 
    Output: precison at given threshold
    '''

    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    Find recall at given threhold 
    Input: Binary outcome from test data & predicted probabilities from applying train on test data 
    Output: recall at given threshold
    '''
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)

    return recall

def baseline(y_test):
    '''
    Find baseline, converted to int type
    Input: Predicted y from train data  
    Output: baselines
    '''

    nr = y_test.astype('int').sum()
    base = nr / len(y_test)

    return base

def accuracy_at_k(y_true, y_scores, k):
    '''
    Find accuracy at given threhold 
    Input: Binary outcome from test data & predicted probabilities from applying train on test data 
    Output: accuracy at given threshold
    '''

    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    pred = accuracy_score(y_true, preds_at_k)

    return pred

def f1_at_k(y_true, y_scores, k):
    '''
    Find F1 (harmonic mean of precison and recall) at given threhold 
    Input: Binary outcome from test data & Predicted probabilities from model on training data 
    Output: F1 at given threshold
    ''' 

    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    f1 = f1_score(y_true, preds_at_k)

    return f1

def compare_metrics(model_result, pivot_by='model_type', metric='precison'):
    '''
    Compare metrics from multiple model across different by different pivot
    Input: model results from the classifier loop, pivot by = time, classifier used, metric
    Output: plot
    ''' 

    # Change name of metric identifier to identify from model result column name
    if metric == 'precision' or metric == 'Precision':
        metric_iden = 'p_'
    elif metric == 'recall' or metric == 'Recall':
        metric_iden = 'r_'
    if metric == 'accuracy' or metric == 'Accuracy':
        metric_iden = 'a_'
    if metric == 'f1' or metric == 'F1' or metric == 'F-1':
        metric_iden = 'f1_'
        
    # Groupby Data by desired outcome
    met_col = [col for col in model_result if col.startswith(metric_iden)]
    d = dict.fromkeys(met_col, ['mean'])
    plot_df = model_result.groupby(pivot_by, as_index=False).agg(d)
    plot_df.columns = plot_df.columns.droplevel(-1)
    plot_df = plot_df.set_index(pivot_by)
    
    # Multiplots for different threholds
    plt.figure(figsize=(8,6))
    for i in range(len(plot_df)):
        plt.plot([k for k in plot_df.columns],[plot_df[y].iloc[i] for y in plot_df.columns])
    plt.legend(plot_df.index,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def pr_curve(y_true, y_prob):
    '''
    Precision Recall Curve for individual y_test and y_prob values using precison recall curve function from sci
    Input: actual outcome from test and predicted probabilities from applying train on test data
    Output: PR plot 
    ''' 

    prec, rec, tre = precision_recall_curve(y_true, y_prob)    
    # Plot 1 axis
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(tre, prec[:-1], 'b--', label='precision')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    # Plot 2 axis
    ax2.plot(tre, rec[:-1], 'r--', label = 'recall')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    plt.show()


def plot_precision_recall_n(y_true, y_score, model_name):
    '''
    Precision Recall Curve across thresholds individual y_test and y_prob values using precison recall curve function from sci
    Input: actual outcome from test and predicted probabilities from applying train on test data and model name
    Output: PR plot 
    ''' 

    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()




# ------------------------------------------------
# Section II: Model Run
# ------------------------------------------------
def define_clfs_params(grid_size):
    '''
    Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    Adapted from magicloops, added njobs to optimise speed 
    '''

    clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'BG': BaggingClassifier(LogisticRegression(penalty='l1', C=1,n_jobs=-1)),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=15,n_jobs=-1),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        #'KNN': KNeighborsClassifier(n_neighbors=3) 
        # Consider kd tree for leaner implementation
        'KNN': KNeighborsClassifier(algorithm='kd_tree',n_jobs=-1)
            }

    small_grid = { 
        'RF':{'n_estimators': [10,100,500], 'max_depth': [1,5,10,20], 'max_features': ['sqrt'], 'n_jobs': [-1]},
        # no bootstrap, bootstrap_features for time to train constraints
        'BG':{'n_estimators': [10,20], 'max_samples': [0.1, 0.5], 'max_features':[0.1, 0.5]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,50,100,500,1000]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.01, 0.1,1]},
        # Same as test row for run time considerations
        'SVM' :{'C' :[0.01,0.1],'kernel':['linear']},
        # Same as test row for time to run considerations
        'GB': {'n_estimators': [10,100], 'learning_rate' : [0.1,0.5],'subsample' : [0.8], 'max_depth': [10,20], 'min_samples_split':[500]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20],'min_samples_split': [2,5,10]},
        'KNN' :{'n_neighbors': [5,10,25],'weights': ['uniform','distance'],'algorithm': ['auto','kd_tree']}
               }
        
    test_grid = { 
        'RF':{'n_estimators': [1,5], 'max_depth': [1,5], 'max_features': ['sqrt'],'min_samples_split': [10]},
        'BG':{'n_estimators': [1,5], 'max_samples': [0.1,0.5], 'max_features':[0.1,0.5]},
        'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1,5]},
        'LR': { 'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]},
        'SVM' :{'C' :[0.01],'kernel':['linear']},
        'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20],'min_samples_split': [2,5,10]},
        'KNN' :{'n_neighbors': [20,25],'weights': ['uniform'],'algorithm': ['kd_tree']}
               }    

    if (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, training_dates, testing_dates):
    '''
    Run lopp over all defined models for 1 set of specification to run and all permutations of respective hyperparameters
    Called from the wrapper function which slpits the data. 
    Input: model to run, classifiers to be used (integrated via dictionary), train & test data, train & test temporal window
    Output: all model performance across all permutations of respective hyperparameters(csv)  
    '''

    results = []
    for n in range(1, 2):
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    start = time.time()
                    # Feature Engeering and Variable Treatment on Training Data
                    model_fit = clf.fit(X_train, y_train)
                    y_pred_probs = model_fit.predict_proba(X_test)[:,1]
                    print('---model fitting--\n')                    
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    row = [training_dates, testing_dates, models_to_run[index],clf, p,
                                                       baseline(y_test.astype('int')),   
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       accuracy_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       ]
                    results.append(row)
                    clfs_time = (time.time() - start) 
                    print('Time =',clfs_time,'for running:\n',clf,'\n')
                except IndexError as e:
                    print('Error:',e)
                    continue
        print('\n\n')
    return results


def clf_wrapper(df, start_time_date, end_time_date, prediction_windows, time_var, pred_var, models_to_run, clfs, grid,\
    missing_var, cat_var, bin_var, norm_var, maj_cities, observation_period):
    '''
    Integrates clf loop and runs the model on different temporal subsets of data
    Obtains results for each model and parameter from from clf_loop row wise : to reduce time
    Input: user spcified start & end window, predicted var (Y) & features, identified classifers and hyperparameters to be run
    Output: all model performance across all permutations of respective hyperparameters(csv)  
    '''

    test_output = []
    # First section adapted from Magicloops used in conjunction with manual/test/train splits function above
    for prediction_window in prediction_windows:
        train_start_time = start_time_date
        train_end_time = train_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+1)
        while train_end_time + relativedelta(months=+prediction_window)<=end_time_date:
            test_start_time = train_end_time + relativedelta(days=+observation_period)
            test_end_time = test_start_time + relativedelta(months=+prediction_window) - relativedelta(days=+1)
            print('training date range:', train_start_time, train_end_time) 
            print('testing date range: \n', test_start_time, test_end_time)

            # Build training and testing sets
            X_train, y_train, X_test, y_test = temporal_train_test_split(df, train_start_time, train_end_time, \
                test_start_time, test_end_time, time_var, pred_var)

            # Run feature engeenering on attributes: imputation, discretization etc
            X_train = feature_eng(X_train, missing_var, cat_var, bin_var, norm_var, maj_cities)
            X_test = feature_eng(X_test, missing_var, cat_var, bin_var, norm_var, maj_cities)
            # Drop extraneous colunms to match test and train data
            common_cols = np.intersect1d(X_test.columns, X_train.columns)
            X_test = X_test[common_cols]
            X_train = X_train[common_cols]

            # Build classifiers: refers to loop identified before for execution
            row_lst = clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, \
                (train_start_time,train_end_time), (test_start_time,test_end_time))
            
            # Add time
            train_end_time += relativedelta(months=+prediction_window)
            test_output.extend(row_lst)
            # Check if Test Data is not Skewed
            print('Check for Skewed Classes')
            sns.countplot(y_test)

    results_df = pd.DataFrame(test_output, columns=('training_dates', 'testing_dates', 'model_type','clf', 
        'parameters', 'baseline', 'auc-roc','a_at_5', 'a_at_20', 'a_at_50', 'f1_at_5', 'f1_at_20', 'f1_at_50', 
        'p_at_1','p_at_5', 'p_at_10', 'p_at_20','p_at_50','r_at_1','r_at_5', 'r_at_10', 'r_at_20','r_at_50'))

    return results_df

def best_model(model_data, model_results, time_var, pred_var, tr_date, te_date, m_var, c_var, b_var, n_var, mj_cities,\
         eval_stat='f1_at_50', window='best'):
    '''
    Extracts and plots Best Model based on eval_stat : AUC/Precison/Recall/F1
    Gives result for best model by default, else by specified period
    Note: requires improvement for inclduing variable importance plots, dual density plots and a tree diagram 
    '''
    # Filter Baseline
    best_mod = model_results[model_results['model_type'] != 'baseline']

    # Model Statistics
    print(best_mod.loc[best_mod[eval_stat].idxmax()].clf)
    mod_stat = best_mod.loc[best_mod[eval_stat].idxmax()]

    # Get Iloc of best model
    bi = best_mod.loc[best_mod[eval_stat].idxmax()].name

    # Extract best model parameters
    if window == 'best':
        best_train = best_mod.loc[bi].training_dates
        best_test = best_mod.loc[bi].testing_dates  
    else: 
        best_train = tr_date
        best_test = te_date

    # Derive train and test sets
    bx_train, by_train, bx_test, by_test = temporal_train_test_split(model_data, best_train[0], best_train[1],best_test[0], 
                                                                                best_test[1], time_var, pred_var)
    # Check Class Skewness
    print('Class Skewness')
    fig = plt.figure(figsize=(6, 4))
    ax=sns.countplot(by_test)
    plt.show()

    # Run feature engeenering on attributes: imputation, discretization etc
    bx_train = feature_eng(bx_train, m_var, c_var, b_var, n_var, mj_cities)
    bx_test = feature_eng(bx_test, m_var, c_var, b_var, n_var, mj_cities)

    # Consistent features
    best_cols = np.intersect1d(bx_test.columns, bx_train.columns)
    bx_test = bx_test[best_cols]
    bx_train = bx_train[best_cols]

    # Fit model
    best = best_mod.loc[bi].clf
    best_fit = best.fit(bx_train, by_train)

    # Generate predictions
    by_pred_probs = best_fit.predict_proba(bx_test)[:,1]

    # Plot precision/recall graph
    plot_precision_recall_n(by_test,by_pred_probs,best)

    return best_fit, mod_stat


# ------------------------------------------------
# Section III: APPENDIX: Improving pipeline further
# ------------------------------------------------
# Var Imp
# feature_imp_all = pd.Series(best_fit.feature_importances_,index=bx_train.columns).sort_values(ascending=False)
# feature_imp = feature_imp_all[0:25]
# # Add labels 
# plt.figure(figsize=(6,4))
# sns.barplot(x=feature_imp, y=feature_imp.index)
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")

# Dual Distribution if Y_pred was also in results
# mod=results.copy(deep=True)
# mod['TF'] = np.where(mod['baseline'] > 0.5, 1, 0)
# g = sns.FacetGrid(mod, col='model_type', hue='TF',palette="Set1")
# g.map(sns.kdeplot, 'baseline', shade=True, label='Data')\
#          .add_legend()\
#          .set_titles("{col_name}")\
#          .set_axis_labels('')


