import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def append_id(data,feature):
    '''
    Takes the dataframe with a list of categories and appends the correct user id to them.

    Parameters
    ----------
    data : dataframe
        Dataframe with a list of categories

    feature : string
        Column name of the data in the json file.

    Returns
    -------
    df_id: dataframe
        A dataframe with added user ids
    '''
    df = []
    for i in data.index: #['user_id'].values:
        df.append(pd.DataFrame(data[feature][i]))
        df[i]['id'] = i

    df_id = pd.concat(df).reset_index(drop=True)
    
    return df_id


def process_json(path,feature):
    '''
    Reads the data from json file and appends the user id to each row.
    Returns the dataframe with the group_name, date_joined and id columns.

    Parameters
    ----------
    path : string
        A path to the json file.

    feature : string
        Column name of the data in the json file.

    Returns
    -------
    df_json: dataframe
        A dataframe with all the data from the json file.
    '''
    with open(path,'r') as f:
        data = json.loads(f.read())
    
    train_json = pd.json_normalize(data, record_path=['data'])
    
    train_json['id'] = train_json['id'].astype(int).values

    df_json = append_id(train_json,feature)
    
    return df_json

def transform_categories(data,feature):
    '''
    Transforms the dataframe with a list of categories and user ids into a one hot encoded 
    dataframe of these categories.
    Returns the dataframe with the one hot encoded categories, indexed by user id.

    Parameters
    ----------
    data : dataframe
        Dataframe with a list of categories and user ids

    feature : string
        Column name of the data in the json file.

    Returns
    -------
    df_json: dataframe
        A dataframe with all the data from the json file.
    '''
    length = data['id'].max() + 1
    id_list = pd.DataFrame(data.groupby(feature)['id'].apply(list))
    groups_cols = pd.DataFrame(index=range(length),columns=id_list.index).fillna(0)

    for group in groups_cols.columns:
        groups_cols[group].loc[id_list['id'].loc[group]] = 1

    return groups_cols


def calculate_percentages(df,ntrain,column,sort=False):
    '''
    Calculates the percentage of people interested in gym memberships for a 
    given categorical column for each category.

    Parameters
    ----------
    df : dataframe
        Dataframe with categorical features

    ntrain : integer
        Number of rows in the training set

    column : string
        Column name of the categorical feature.

    sort : boolean
        Boolean value indicating whether to sort the results

    Returns
    -------
    df_percentages: dataframe
        A dataframe interest percentages for each category in the given feature
    '''
    interested = df[df['target']==1][[column,'user_id']].groupby(column).count()
    total = df[['target',column]][:ntrain].groupby([column]).count()
    percentages = np.round((interested.values/total.values)*100,0)
    
    df_percentages = pd.DataFrame(index=total.index,data=percentages,columns=['Percentage of interested'])
    if sort:
        df_percentages = df_percentages.sort_values(by='Percentage of interested',ascending=False)
    return df_percentages


def calculate_correlation(data,target_df,target='target'):
    '''
    Calculates the correlations between given features and the target variable 
    and returns them as a dataframe.

    Parameters
    ----------
    data : dataframe
        Dataframe with features that we want to calculate correlation with.

    target_df : dataframe
        Dataframe containg the target variable.

    target : string
        Column name of the taget variable for calculating correlation.

    Returns
    -------
    df_corr: dataframe
        A dataframe with correlation coefficients between the given featues and target variable.
    '''
    length = len(data)
    correlations = []
    for col in data.columns:
        correlations.append(pd.DataFrame(target_df[target]).corrwith(data[col],axis=0))

    df_corr = pd.DataFrame(correlations).set_index(data.columns).sort_values(by=target,ascending=False)
    df_corr.columns = ['Correlation with target']

    return df_corr


def nan_count(data,column):
    '''
    Calculates the amount of missing values and print out the results.

    Parameters
    ----------
    data : dataframe
        The dataframe containg the columns with missing values.

    column : string
        Name of the feature to check for null values.

    Returns
    -------
    
    '''
    missing_data = data[column].isnull()
    print(column)
    print(missing_data.value_counts())
    print('\n')


def plot_heatmap(df,feats,size=(12, 8)):
    '''
    Calculates the correlations between given features and the target variable 
    and returns them as a dataframe.

    Parameters
    ----------
    df : dataframe
        Dataframe with features that we want to plot correlations for.

    feats : string
        Names of the features to plot in the heatmap.

    size : tuple
        The tuple of two integers indicating the size of the heatmap.

    Returns
    -------

    '''
    fig, ax = plt.subplots(1, 1, figsize=size)
    sns.heatmap(df[feats].corr(), annot = True, ax=ax)


def draw_histplots(data, feature, rotate_labels=False, rot=45, xlabel=None, ylabel=None,bins=10,legend_labels=None):
    '''
    Draws two histograms for the given feature, one normal and one with data separated by the target variable
    and calculates and prints out the mean and standard deviation of these histograms.

    Parameters
    ----------
    data : dataframe
        The dataframe containg the columns to plot.

    feature : string
        Name of the feature to use for the histograms.

    rotate_labels : boolean
        Boolean value indicating whether to rotate the x labels or not.

    rot : integer
        Number indicating the angle for x label rotation.

    xlabel,ylabel : string
        Names for the x and y axes.

    bins : integer
        The number of bins to plot on the histograms.  

    legend_labels : list
        List of string with the new labels for the legend.
   
    Returns
    -------
    chart1: AxesSubplot
        A subplot containg a histogram of the selected feature.

    chart2: AxesSubplot
        A subplot containg a histogram of the selected feature separated by the target variable.
   
    '''
    sns.set(font_scale=1.3)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    chart1 = sns.histplot(x=feature,data=data, ax=axes[0],bins=bins)
    chart2 = sns.histplot(x=feature,data=data, hue='target', ax=axes[1],bins=bins)

    if legend_labels:
        chart2.legend(legend_labels)
    
    if rotate_labels:
        chart1.set_xticklabels(labels = chart1.get_xticklabels(),rotation=rot, horizontalalignment='right')
        chart2.set_xticklabels(labels = chart2.get_xticklabels(),rotation=rot, horizontalalignment='right')
    
    if xlabel:   
        for ax in axes:
            ax.set(xlabel=xlabel)
        
    if ylabel:   
        for ax in axes:
            ax.set(ylabel=ylabel)
    plt.tight_layout()
    
    mean_total = np.round(data[feature].mean(),0)
    std_total = np.round(data[feature].std(),0)
    mean_interested = np.round(data[data['target'] == 1][feature].mean())
    std_interested = np.round(data[data['target'] == 1][feature].std())
    mean_not_interested = np.round(data[data['target'] == 0][feature].mean())
    std_not_interested = np.round(data[data['target'] == 0][feature].std())
    
    print(f'All of the people: mean {feature}: {mean_total}, standard deviation: {std_total}')
    print(f'The people interested in gym: mean {feature}: {mean_interested}, standard deviation: {std_interested}')
    print(f'The people not interested in gym: mean {feature}: {mean_not_interested}, standard deviation: {std_not_interested}')

    return chart1,chart2


def make_crosstable(data,column,target='target'):
    '''
    Creates a crosstable of the selected feature and the target variable.

    Parameters
    ----------
    data : dataframe
        The dataframe containg the columns to plot.

    column: string
        Names of the feature to use in a crosstable.

    target : string
        Name of the target variable.

    Returns
    -------
        
    crosstab : dataframe
        A crosstable created for the selected features.
    '''
    crosstab = pd.crosstab(data[column][:ntrain],data[target][:ntrain],margins=True)
    crosstab = crosstab.style.background_gradient(cmap='Blues')
    return crosstab


def draw_countplots(data,feature, target='target',plot_number=2, rotate_labels=False, rot=45, xlabel=None, ylabel=None, legend_labels=None, legend_loc=0,tick_labels=None):
    '''
    Draws two countplots of the selected feature, one normal and one with data separated by the target variable

    Parameters
    ----------
    data : dataframe
        The dataframe containg the columns to plot.

    feature : string
        Name of the feature to use for the countplots.

    target : string
        Name of the feature to use as hue in the second plot.

    plot_number : integer
        An integer value (1 or 2) that indicates how many plots should be created.

    rotate_labels : boolean
        Boolean value indicating whether to rotate the x labels or not.

    rot : integer
        Number indicating the angle for x label rotation.

    xlabel,ylabel : string
        Names for the x and y axes.

    bins : integer
        The number of bins to plot on the histograms.  

    legend_labels : list
        List of string with the new labels for the legend.

    legend_loc : integer or string
        Integer indicating where should the legend be positioned. Following values are accepted:

        Location String   Location Code
        ===============   =============
        'best'            0
        'upper right'     1
        'upper left'      2
        'lower left'      3
        'lower right'     4
        'right'           5
        'center left'     6
        'center right'    7
        'lower center'    8
        'upper center'    9
        'center'          10

    tick_labels : list
        List of string to replace the original x axis tick labels.
   
    Returns
    -------
    chart1: AxesSubplot
        A subplot containg a countplot of the selected feature.

    chart2: AxesSubplot
        A subplot containg a countplot of the selected feature separated by the target variable.
   
    '''
    sns.set(font_scale=1.3)
    fig, axes = plt.subplots(1, plot_number, figsize=(12, 8))
    
    if plot_number == 2:
        chart1 = sns.countplot(x=feature,data=data, ax=axes[0]) 
        chart2 = sns.countplot(x=feature,data=data, hue=target, ax=axes[1])  
    else:
        chart2 = sns.countplot(x=feature,data=data, hue=target, ax=axes)
    
    if legend_labels:
        chart2.legend(legend_labels,loc=legend_loc)
    
    if tick_labels:
        if plot_number == 2:
            chart1.set_xticklabels(tick_labels)
        chart2.set_xticklabels(tick_labels)
    
    if rotate_labels:
        if plot_number == 2:
            chart1.set_xticklabels(labels = chart1.get_xticklabels(),rotation=rot, horizontalalignment='right')
        chart2.set_xticklabels(labels = chart2.get_xticklabels(),rotation=rot, horizontalalignment='right')
    
    if xlabel:   
        for ax in axes:
            ax.set(xlabel=xlabel)
        
    if ylabel:   
        for ax in axes:
            ax.set(ylabel=ylabel)
    plt.tight_layout()
    plt.show()
    
    if plot_number == 2:
        return chart1,chart2
    else:
        return chart2


def draw_catplot(data,x,hue,y='target',rot=None, legend_labels=None):
    '''
    Draws a categorical plot based on provided feature labels.

    Parameters
    ----------
    data : dataframe
        The dataframe containg the columns to plot.

    x : string
        Name of the feature to plot on the x axis.

    hue : string
        Name of the categorical feature for separating the plots.

    y : string
        Name of the feature to plot on the y axis.

    rot : integer
        Number indicating the angle for x label rotation.

    legend_labels : list
        List of string with the new labels for the legend.

    Returns
    -------
    chart: AxesSubplot
        A subplot containg the categorical plot of the selected features.
    '''
    chart = sns.catplot(data=data, x=x, y=y, hue=hue, kind='point',legend_out=False)
    if rot:
        chart.set_xticklabels(rotation=rot, horizontalalignment='right')
    if legend_labels:
        for t,l in zip(chart._legend.texts,legend_labels): t.set_text(l)

    plt.show()
    
    return chart

def model_assessment(X_train,y_train,MLA):
    '''
    Runs cross_validation on the provided models and creates a table with scores for each one.

    Parameters
    ----------
    X_train : dataframe
        The dataframe containg the features from the training set.

    y_train : series
        The pandas series containg the target variable.

    MLA : list
        A list containg the models to be tested.

    Returns
    -------
    MLA_compare: dataframe
        Dataframe with names, parameters, f1 scores for each model.
    '''
    cv_split = ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 ) # run model 10x with 80/20 split (because of small amount of data) 

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train f1 Score Mean', 'MLA Test f1 Score Mean' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = y_train.copy()


    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation
        cv_results = cross_validate(alg, X_train, y_train, cv=cv_split, scoring='f1', return_train_score=True)

        MLA_compare.loc[row_index, 'MLA Train f1 Score Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test f1 Score Mean'] = cv_results['test_score'].mean()   

        row_index+=1

    MLA_compare.sort_values(by = ['MLA Test f1 Score Mean'], ascending = False, inplace = True) 
    
    return MLA_compare

def get_predictions(clf, train, test, features, target_col, N_SPLITS = 5,scaling=False,fi_enable = False,fit_params={}):
    '''
    A function that performs the stratified cross validation of the provided ML model and returns 
    the predictions of the test set as well as the out of fold predictions to use later for ensemble learning.

    Parameters
    ----------
    clf : model
        The classification model to use for fitting and predicting.

    train : dataframe
        The dataframe containing the training set

    test : dataframe
        The dataframe containing the test set

    features : list
        The list of features selected for modelling.

    target_col : string
        The name of the target variable to predict.

    N_SPLITS : integer
        Number of splits to make for cross validation.

    scaling : boolean
        Boolean value that enable the scaling of the features.

    fi_enable : boolean
        Boolean value that enables the calculation of feature importance (not all models provide this)

    fit_params : dictionary
        The dictionary containggg additional parameters for fitting,
        for example: fit_params = {'verbose': 300, 'early_stopping_rounds': 200}.

    Returns
    -------
    oofs: array
        An array containg the out of fold predictions.

    preds: array
        An array containg the test set predictions.
    '''
  
    #oof - out of fold
    train_preds = np.zeros(len(train))
    oofs = np.zeros(len(train))
    preds = np.zeros(len(test))

    target = target_col
    
    # separating the target variable and the selected features
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]

    if fi_enable == True:
        feature_importances = pd.DataFrame()
    
    skf = StratifiedKFold(n_splits=N_SPLITS)


    for fold_, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f'\n------------- Fold {fold_ + 1} -------------')

        ### Training Set
        X_trn, y_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]

        ### Validation Set
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        
        if scaling == True:
            scaler = StandardScaler()
            scaler.fit(X_trn)

            X_trn = scaler.transform(X_trn)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        if fit_params:
            clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)], **fit_params)
        else:
            clf.fit(X_trn, y_trn)

        if fi_enable == True:
            fold_importance = pd.DataFrame({'fold': fold_ + 1, 'feature': features, 'importance': clf.feature_importances_})
            feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        ### Instead of directly predicting the classes we will obtain the probability of positive class.
        preds_train = clf.predict(X_trn)
        preds_val = clf.predict(X_val)
        preds_test = clf.predict(X_test)

        train_score = f1_score(y_trn, preds_train)
        fold_score = f1_score(y_val, preds_val)
        print(f'\nf1 score for train set is {train_score}')
        print(f'\nf1 score for validation set is {fold_score}')
        
        train_preds[trn_idx] = preds_train
        oofs[val_idx] = preds_val
        preds += preds_test / N_SPLITS

    train_preds_score = f1_score(y_train, train_preds)
    oofs_score = f1_score(y_train, oofs)
    print('\nTotal f1 scores:')
    print(f'\nf1 score for train set is {train_preds_score}')
    print(f'\nf1 score for oofs is {oofs_score}')
    
    if fi_enable == True:
        feature_importances = feature_importances.reset_index(drop = True)
        fi = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending = False)[:20][::-1]
        fi.plot(kind = 'barh', figsize=(12, 6))
        return oofs, preds, fi
    
    return oofs, preds

def submit_file(final_results, filename='test.csv'): 
    '''
    Saves the dataframe to a csv file.

    Parameters
    ----------
    final_results : dataframe
        The dataframe containg the user_id, propensity and the predictions

    filename : string
        The name for the csv file.

    Returns
    -------
    '''
    final_results.to_csv(filename, index=False)