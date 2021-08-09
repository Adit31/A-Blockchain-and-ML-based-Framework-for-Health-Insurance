"""Premium_Final.ipynb
-*- coding: utf-8 -*-
Author: Adit Goyal
Date: 05/08/2021
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('insurance.csv')

df_reg = df.select_dtypes(include=['object']).copy()

labels_region = df_reg['region'].astype('category').cat.categories.tolist()
labels_smoker = df_reg['smoker'].astype('category').cat.categories.tolist()
labels_sex = df_reg['sex'].astype('category').cat.categories.tolist()

replace_region_comp = {'region' : {k: v for k,v in zip(labels_region,list(range(1,len(labels_region)+1)))}}
replace_smoker_comp = {'smoker' : {k: v for k,v in zip(labels_smoker,list(range(1,len(labels_smoker)+1)))}}
replace_sex_comp = {'sex' : {k: v for k,v in zip(labels_sex,list(range(1,len(labels_sex)+1)))}}

print(replace_region_comp)
print(replace_smoker_comp)
print(replace_sex_comp)

df_reg_replace=df_reg.copy()

df_reg_replace.replace(replace_region_comp, inplace=True)
df_reg_replace.replace(replace_smoker_comp, inplace=True)
df_reg_replace.replace(replace_sex_comp, inplace=True)

df=df.drop(['sex', 'smoker', 'region'],axis=1)

df=df.join(df_reg_replace)

#Normalizing
normalized_df = df
normalized_df[df.columns] = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df[df.columns]))

X=normalized_df.drop(columns=['charges'])
y = normalized_df.iloc[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#INTERACTION TERMS
poly = PolynomialFeatures(degree=7, interaction_only = False, include_bias = True)
X_poly_train=poly.fit_transform(X_train)
X_poly_test=poly.fit_transform(X_test)

#Ridge Regression
ridge=Ridge()
parameters={'alpha':[1e-2, 3e-2, 0.07, 0.09, 1e-1, 1, 3, 4]}
ridge_regressor=GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv = 3)
ridge_regressor.fit(X_poly_train,y_train)

print(ridge_regressor.best_score_)
print(ridge_regressor.best_params_)

#DistPlot - Train

prediction_ridge=ridge_regressor.predict(X_poly_train)

sns.distplot(y_train-prediction_ridge, axlabel='Premiums')

#DistPlot - Test
prediction_ridge=ridge_regressor.predict(X_poly_test)

sns.distplot(y_test-prediction_ridge, axlabel='Premiums')

#Train, Test scores
predict_train = ridge_regressor.predict(X_poly_train)
predict_test = prediction_ridge
r2scores=[r2_score(y_train, predict_train), r2_score(y_test, predict_test)]
rmse=[mean_squared_error(y_train, predict_train, squared = False), 
      mean_squared_error(y_test, predict_test, squared = False)]
mae=[mean_absolute_error(y_train, predict_train), mean_absolute_error(y_test, predict_test)]

dataset = pd.DataFrame(list(zip(r2scores,rmse,mae)), columns=['R2','RMSE','MAE'])

index = np.arange(2)
sets=['Train', 'Test']
bar_width = 0.25
score_label = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots()
barR2 = ax.bar(index-bar_width, r2scores, bar_width, label='R2', color = 'mediumseagreen')
barRMSE = ax.bar(index, rmse, bar_width, label='RMSE', color = 'lightcoral')
barMAE = ax.bar(index+bar_width, mae, bar_width, label='MAE', color = 'cornflowerblue')
ax.set_xticks(index)
ax.set_xticklabels(sets)
ax.set_yticks(score_label)
ax.legend()

def insert_data_labels(bars):
    for bar in bars:
        bar_height=bar.get_height()
        ax.annotate('{:.4f}'.format(bar.get_height()), 
                    xy=(bar.get_x() + bar.get_width()/2, bar_height), 
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    va='bottom')

insert_data_labels(barR2)
insert_data_labels(barRMSE)
insert_data_labels(barMAE)

plt.show()
