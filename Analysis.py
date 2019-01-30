# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 02:04:58 2019

@author: David Delgado
"""

import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()

data_train1 = pd.read_csv('./CMAPSSData/train_FD001.txt', sep = ' ', header = None)
data_test1 = pd.read_csv('./CMAPSSData/test_FD001.txt', sep = ' ', header = None)
data_RUL1 = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep = ' ', header = None)


unit_1 = data_train1[data_train1[0] == 1]

#%% EDA of what each sensor or setting looks like 
plt.close('all')
#plot all the sensor data
plt.plot(unit_1[1], unit_1.loc[:,5:28], 'o' )
plt.show()
#seems like theres quite the spread

#%% Scale the sensor data in a meaningful way 
