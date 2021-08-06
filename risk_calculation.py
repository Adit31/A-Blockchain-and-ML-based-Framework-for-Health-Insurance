"""
-*- coding: utf-8 -*-
Author: Adit Goyal
Date: 07/08/2021
"""

import pandas as pd
import numpy as np
from numpy.random import *
from ml_metrics import quadratic_weighted_kappa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from adjustText import adjust_text
import matplotlib.pyplot as plt

def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)

def score_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

def apply_offsets(data, offsets):
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j]
    return data

# global variables
columns_to_drop = ['Id', 'Response']
missing_indicator = -1000

all_data = pd.read_csv("train.csv")

# create any new variables    
all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

all_data.fillna(missing_indicator, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

train, test = train_test_split(all_data, test_size = 0.1, random_state = 752)

#Predictor and responce variables
train_x = train.drop(['Id', 'Response'], axis = 1)
train_y = train['Response']
test_x = test.drop(['Id', 'Response'], axis = 1)
test_y = test['Response']

#Function for normalization
def normalization(data):
    return (data - data.min())/(data.max() - data.min())

#Model for the Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, max_features=75, max_depth=50, 
                               min_samples_leaf=8, min_samples_split=40,
                               bootstrap = False)

model.fit(train_x, train_y)

test_pred = model.predict(test_x) 
train_pred = model.predict(train_x)

print("Classification score for testing set: ", classification_report(test_y, test_pred))
print("Classification score for training set: ", classification_report(train_y, train_pred))

#Accuracy-training set
print(accuracy_score(train_y, train_pred))
print(f1_score(train_y, train_pred, average = 'weighted'))
print(precision_score(train_y, train_pred, average = 'weighted'))
print(recall_score(train_y, train_pred, average = 'weighted'))

#Accuracy-testing set
print(accuracy_score(test_y, test_pred))
print(f1_score(test_y, test_pred, average = 'weighted'))
print(precision_score(test_y, test_pred, average = 'weighted'))
print(recall_score(test_y, test_pred, average = 'weighted'))

#DistPlot - Train
sns.distplot(abs(np.array(train_y)-train_pred), kde=False, axlabel='Classification Error (Ground truth - Predicted Value)')

#DistPlot - Test
sns.distplot(abs(test_y-test_pred), kde=False, axlabel='Classification Error (Ground truth - Predicted Value)')

# RMSE
print(np.sqrt(mean_squared_error(test_y, test_pred)))
print(np.sqrt(mean_squared_error(train_y, train_pred)))

# MAE
print(mean_absolute_error(test_y, test_pred))
print(mean_absolute_error(train_y, train_pred))

#Quadratic Weighted Kappa
print(eval_wrapper(train_pred, train_y))
print(eval_wrapper(test_pred, test_y))

rmse = [2.031708524115556, 2.263981089169368]
mae = [0.9225702630889563, 1.1885839366896784]
acc = [0.7069159088357472, 0.5861256103721165]
qwk = [0.6144943279531684, 0.5027416257338551]

index = np.arange(2)
sets = ['Train','Test']
bar_width = 0.18
score_label = np.arange(0, 4, 0.4)
fig, ax = plt.subplots()
barAcc = ax.bar(index - 1.5*bar_width, acc, bar_width, label = 'Accuracy', color = 'darkcyan')
barRMSE = ax.bar(index - 0.5*bar_width, rmse, bar_width, label = 'RMSE', color = 'salmon')
barMAE = ax.bar(index + 0.5*bar_width, mae, bar_width, label = 'MAE', color = 'seagreen')
barQWK = ax.bar(index + 1.5*bar_width, qwk, bar_width, label = 'Quadratic Weighted Kappa', color = 'gold')
ax.set_xticks(index)
ax.set_xticklabels(sets)
ax.set_yticks(score_label)
ax.legend(loc="upper left")

def insert_data_labels(bars):
    for bar in bars:
        bar_height=bar.get_height()
        ax.annotate('{:.4f}'.format(bar.get_height()), 
                    xy=(bar.get_x() + bar.get_width()/2, bar_height), xytext = (0, 3),
                    textcoords = 'offset points', ha = 'center', va = 'bottom')

insert_data_labels(barAcc)
insert_data_labels(barRMSE)
insert_data_labels(barMAE)
insert_data_labels(barQWK)

plt.show()

def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height:
                differ = np.diff(sorted_ltp, axis = 0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2:
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - txt_width, 1.01*t, '%d'%int(y), rotation = 0, color = 'blue')
        if y != t:
            axis.arrow(x, t, 0, y-t, color = 'red', alpha = 0.3, width = txt_width*0.1, 
                       head_width = txt_width, head_length = txt_height*0.5, 
                       zorder = 0, length_includes_head = True)

f1_1 =[0.50, 0.23]
f1_2 = [0.51, 0.29]
f1_3 = [0.58, 0.49]
f1_4 = [0.72, 0.66]
f1_5 = [0.67, 0.59]
f1_6 = [0.70, 0.54]
f1_7 = [0.66, 0.45]
f1_8 = [0.82, 0.78]
index = np.arange(2)
sets = ['Train', 'Test']
bar_width = 0.12
score_label = np.arange(0, 1, 0.1)
fig, ax = plt.subplots()
bar1 = ax.bar(index - 3.5*bar_width, f1_1, bar_width, label = 'Class 1', color = 'lightcoral')
bar2 = ax.bar(index - 2.5*bar_width, f1_2, bar_width, label = 'Class 2', color = 'bisque')
bar3 = ax.bar(index - 1.5*bar_width, f1_3, bar_width, label = 'Class 3', color = 'gold')
bar4 = ax.bar(index - 0.5*bar_width, f1_4, bar_width, label = 'Class 4', color = 'lightgreen')
bar5 = ax.bar(index + 0.5*bar_width, f1_5, bar_width, label = 'Class 5', color = 'teal')
bar6 = ax.bar(index + 1.5*bar_width, f1_6, bar_width, label = 'Class 6', color = 'royalblue')
bar7 = ax.bar(index + 2.5*bar_width, f1_7, bar_width, label = 'Class 7', color = 'mediumpurple')
bar8 = ax.bar(index + 3.5*bar_width, f1_8, bar_width, label = 'Class 8', color = 'grey')
ax.set_xticks(index)
ax.set_xticklabels(sets)
ax.set_yticks(score_label)
ax.legend(title = 'F1 Scores', loc = "upper right", bbox_to_anchor = (1.25, 1))

def insert_data_labels(bars):
    for bar in bars:
        bar_height = bar.get_height()
        ax.annotate('{:.2f}'.format(bar.get_height()), 
                    xy = (bar.get_x() + bar.get_width()/2, bar_height), 
                    xytext = (0, 3), weight = 'bold', textcoords = 'offset points', 
                    ha = 'center', va = 'bottom').set_fontsize(8)
insert_data_labels(bar1)
insert_data_labels(bar2)
insert_data_labels(bar3)
insert_data_labels(bar4)
insert_data_labels(bar5)
insert_data_labels(bar6)
insert_data_labels(bar7)
insert_data_labels(bar8)

plt.show()