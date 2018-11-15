# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:37:04 2018

@author: Ankita
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns

def merge(df1, df2):
    return pd.concat([df1, df2])

def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

def get_categorial_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def transform_cat_to_cont(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature] = None
        df[feature] = df[feature].map(mappings)

def split(df, ind):
    return(df[0:ind],df[ind:])

def get_features_missing_data(df, cutoff):
    total_missing = df.isnull().sum()
   # n = df.shape[0]
    to_delete = total_missing[(total_missing)> cutoff]
    return list(to_delete.index)

def filter_features(df, features):
    df.drop(features, axis = 1, inplace = True)
    
def viz_cont(df, features):
    for feature in features:
        sns.displot(df[feature], kde=False)
        
def viz_cont_cont(df, features, target):
    for feature in features:
        sns.jointplot(x= feature, y= target, data= df)
        
def viz_cat_cont(df, features, target):
    for feature in features:
        sns.boxplot(x= feature, y= target, data= df)
        plt.xticks(rotation = 45)
    
os.getcwd()
os.environ['PATH'] = os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

house_train = pd.read_csv('C:\\Data Science\\data\\House_Prices\\train.csv')
house_train.shape
house_train.info()

house_test = pd.read_csv('C:\\Data Science\\data\\House_Prices\\test.csv')
house_test.shape
house_test.info()
house_test['SalePrice'] = 0 

house_data = merge(house_train, house_test)
house_data.shape
house_data.info()

print(get_continuous_columns(house_data))
print(get_categorial_columns(house_data))

features = ['MSSubClass']
transform_cont_to_cat(house_data, features)

ordinal_features = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
quality_dict = {'NA': 0, 'Po': 1, 'Fa': 2, 'Ta': 3, 'Gd': 4, 'Ex': 5}
transform_cat_to_cont(house_data, ordinal_features, quality_dict)

print(get_continuous_columns(house_data))
print(get_categorial_columns(house_data))

house_train, house_test = split(house_data, house_train.shape[0])

missing_features = get_features_missing_data(house_train, 0)
filter_features(house_train, missing_features)
house_train.shape
house_train.info()

house_train['log]_sale_price'] = np.log(house_train['SalePrice'])
features =['SalePrice','log_sale_price']
viz_cont(house_train, features)






