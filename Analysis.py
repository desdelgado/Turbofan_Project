# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 02:04:58 2019

@author: David Delgado
"""

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

feat_names = ['Unit', 'CycleNo', 'opset1', 'opset2', 'opset3', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
              'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
              'sensor15', 'sensor16', 'sensor17','sensor18', 'sensor19','sensor20', 'sensor21','sensor22', 'sensor23',
              'sensor24', 'sensor25', 'sensor26']  #Add pythonic regilar expression

data_train1 = pd.read_csv('./CMAPSSData/train_FD001.txt', sep = ' ', header = None, names=feat_names)
# data_test1 = pd.read_csv('./CMAPSSData/test_FD001.txt', sep = ' ', header = None)
# data_RUL1 = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep = ' ', header = None)

# print(data_train1['Unit'].value_counts())

# plt.plot(data_train1['Unit'].value_counts()) # no of data points per unit
# plt.show()

max_cycle = data_train1.groupby('Unit')['CycleNo'].max().reset_index()

# print(max_cycle.shape)
# Add checking for NaNs in all columns

# data_train1['RUL'] =

# for x in max_cycle[0]:

# for x in max_cycle[0]:
#     for k in data_train1['Unit']:
#         if x == data_train1['Unit']:


# unit_1 = data_train1[data_train1[0] == 1]

# EDA of what each sensor or setting looks like
# plt.close('all')
#plot all the sensor data
# plt.plot(unit_1[1], unit_1.loc[:,5:28], 'o' )
# plt.show()
#seems like theres quite the spread

#%% Scale the sensor data in a meaningful way 
