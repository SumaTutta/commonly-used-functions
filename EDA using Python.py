#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:38:35 2017

@author: suma
"""

""" EDA using Python"""


import pandas as pd
import os
"""Terminal:  pip install missingno"""
import missingno as msno
"""Terminal:  pip install sns"""
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import math
""" Terminal: pip install pandas_profilingv"""
import pandas_profiling

os.chdir('/Users/suma/Documents/Data Science/Assignments/Assignment 1/Problem 2')

df_train = pd.read_csv('Ghouls_Train.csv')
df_train.shape
df_train.head()
df_train.describe().T
df_train.dtypes

all_columns = df_train.columns.tolist()

df_train.groupby('type').agg({col: 'nunique' for col in all_columns})
df_train.groupby('color').agg({col: 'nunique' for col in all_columns})

def get_subgroup(dataframe, g_index, g_columns):
    """Helper function that creates a sub-table from the columns and runs a quick uniqueness test"""
    g = dataframe.groupby(g_index).agg({col: 'nunique' for col in g_columns})
    if g[g>1].dropna().shape[0] != 0:
        print("Warning: you probably assumed this had all unique values but it doesn't.")
    return dataframe.groupby(g_index).agg({col: 'size' for col in g_columns})
       
 
test  =  get_subgroup(df_train, 'type', all_columns) 

df_train['color'].value_counts()
df_train['color'].nunique()
df_train['type'].nunique() 

rename_columns = {'color': 'colour'}
df_train = df_train.rename(columns = rename_columns)
df_train.head()


os.chdir('/Users/suma/Documents/Data Science/Titanic Problem/')

titanic_train = pd.read_csv('titanic_train.csv')
msno.matrix(titanic_train.sample(500), figsize = (14, 4), width_ratios=(15,1))

msno.bar(titanic_train.sample(500), figsize = (14, 4),)
msno.heatmap(titanic_train.sample(500), figsize = (14, 4),)

len(titanic_train[titanic_train.Age.isnull()])

df_train['bone_length'].corr(df_train['hair_length'])
df_train['bone_length'].corr(df_train['has_soul'])
df_train['hair_length'].corr(df_train['has_soul'])
df_train['bone_length'].corr(df_train['rotting_flesh'])

df_train.corr()
fig, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(pd.crosstab(df_train.type, df_train.color), cmap = 'Blues', annot = True, fmt = 'd', ax=ax,
                square = True)
ax.set_title("Correlation between Bone Length and Hair Length")
fig.tight_layout()

sns.distplot(df_train.bone_length, kde = False)
sns.distplot(df_train.rotting_flesh, kde = False)
sns.distplot(df_train.hair_length, kde = False)
sns.distplot(df_train.has_soul, kde = False)

sns.distplot(df_train.has_soul)

fig, ax = plt.subplots(figsize = (12, 8))
scatter_matrix(df_train[['hair_length', 'bone_length', 'has_soul', 'rotting_flesh']], alpha = 0.2, diagonal = 'hist', ax = ax)

df_train['GhostLife'] = df_train.loc[0:, ['hair_length', 'bone_length', 'has_soul']].mean(axis = 1)

fig, ax = plt.subplots(figsize = (12, 8))
scatter_matrix(df_train[['GhostLife', 'rotting_flesh']], alpha = 0.2, diagonal = 'hist', ax = ax)

df_train['rotting_tranformed'] = list(map(lambda x: math.sqrt(x), df_train['rotting_flesh']))
df_train.info()
df_train.describe()
sns.distplot(df_train.rotting_tranformed, kde = False)


bins1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
bins2 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
df_train['rotting_binned'] = pd.cut(df_train['rotting_tranformed'], bins2)

bin_categories = ["bin1", "bin2", "bin3", "bin4", "bin5", "bin6", "bin7", "bin8", "bin9", "bin10"]
df_train['GhostLife_binned'] = pd.qcut(df_train['GhostLife'], len(bin_categories), bin_categories)


bin_categories = ["bin1", "bin2", "bin3", "bin4", "bin5", "bin6", "bin7", "bin8", "bin9", "bin10"]
df_train['rotting_binned'] = pd.qcut(df_train['rotting_tranformed'], len(bin_categories), bin_categories)

fig, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(pd.crosstab(df_train.GhostLife_binned, df_train.rotting_binned), cmap = 'Blues', annot = True, fmt = 'd', ax=ax,
                square = True)
ax.set_title("Correlation between type and rotting_tranformed")
fig.tight_layout()


fig, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(pd.crosstab(df_train.type, df_train.rotting_binned), cmap = 'Blues', annot = True, fmt = 'd', ax=ax,
                square = True)
ax.set_title("Correlation between type and rotting_tranformed")
fig.tight_layout()

fig, ax = plt.subplots(figsize = (12, 8))
scatter_matrix(df_train[['GhostLife', 'rotting_tranformed']], alpha = 0.2, diagonal = 'hist', ax = ax)

fig, ax = plt.subplots(figsize = (12, 8))
scatter_matrix(titanic_train, alpha = 0.2, diagonal = 'hist', ax = ax)

fig, ax = plt.subplots(figsize = (12, 8))
sns.regplot('GhostLife', 'rotting_flesh', data = df_train, ax = ax)
ax.set_ylabel("Ghost Life Metric")
ax.set_xlabel("rotting_flesh")
fig.tight_layout

pandas_profiling.ProfileReport(df_train)




