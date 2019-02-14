# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 00:10:52 2019

@author: David Delgado
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 02:04:58 2019

@author: David Delgado
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import seaborn as sns; sns.set()
import time
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings

# from sklearn.preprocessing import StandardScaler, MinMaxScaler

feat_names = ['Unit', 'CycleNo', 'opset1', 'opset2', 'opset3', 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
              'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13', 'sensor14',
              'sensor15', 'sensor16', 'sensor17','sensor18', 'sensor19','sensor20', 'sensor21','sensor22', 'sensor23',
              'sensor24', 'sensor25', 'sensor26']  #Add pythonic regilar expression

data_train1 = pd.read_csv('./CMAPSSData/train_FD001.txt', sep = ' ', header = None, names=feat_names)
#%%
# data_test1 = pd.read_csv('./CMAPSSData/test_FD001.txt', sep = ' ', header = None)
# data_RUL1 = pd.read_csv('./CMAPSSData/RUL_FD001.txt', sep = ' ', header = None)

# data_train1.isna().sum() # Finding NaNs

# print(data_train1['Unit'].value_counts())

# plt.plot(data_train1['Unit'].value_counts()) # no of data points per unit
# plt.show()
#%%Prints out each sensor data 
'''
for column in data_train1.columns:
    plt.figure()
    plt.plot(data_train1['CycleNo'], data_train1[column], 'o')
    plt.show()
'''
#%%
data_train1['RUL'] = ""

max_cycle = data_train1.groupby('Unit')['CycleNo'].max().reset_index()
#Get a corrlation dataframe just to get a feel for what has the highest correlation
#data_train1.corr(method = 'pearson')
df1 = data_train1.drop(['RUL','sensor22','sensor23','sensor25','sensor26','sensor24','Unit','CycleNo'], axis=1)
'''
corr=df1.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');
'''
# print(max_cycle)
# print(max_cycle.loc[[1]])


# Add checking for NaNs in all columns

# print(max_cycle.iloc[2, 1])

#%%
for x in max_cycle.iloc[:, 0]:
    count_cycle = max_cycle.iloc[x-1, 1]
    data_train1.loc[data_train1['Unit'] == x, 'RUL'] = count_cycle - data_train1['CycleNo']

df = data_train1.drop(['RUL','sensor22','sensor23','sensor25','sensor26','sensor24','Unit','CycleNo'], axis=1)
#%% We want to normalize because that reduces the outliers of the 
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.columns = df.columns

#%% Make a function that we can quickly test models without having to use copy and paste.  This will help us get an general idea
#for time and accuracy 
'''
def TestModel(x_data, y_data, model):
    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=5)
    
    t_model = model
    t_model.fit(X_train, y_train)
    predictions = t_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(str(model))
    print('RMSE = {}'.format(round(mse**(0.5), 3)))
    print('Time: ' + str(time.time()-start))
#%% Iterate through different Models

diff_models = [LinearRegression(), RandomForestRegressor(), DecisionTreeRegressor()]
for test in diff_models:
    TestModel(df_normalized, data_train1["RUL"], test)
'''
#%% Iterate through different models
#Toggle this on and off for warnings when we write it up can say: Let this on when developing but turned it off cause SVR was giving 
#us "Future warnings" which have no effect on the quick look 
warnings.filterwarnings("ignore")

num_folds = 7
seed = 5
 
X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_train1["RUL"], test_size=0.3, random_state=5)
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('Bagging', RandomForestRegressor()))

results = []
names = []
times = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    start = time.time()
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring= 'neg_mean_squared_error')
    run_time = time.time()-start
    results.append(cv_results)
    names.append(name)
    times.append(run_time)
    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), run_time)
    print(msg)
#%% Create Visual for people   
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()    
# Add a plot of MSE vs Time labeled with different regression algorithums when we come back 
fig1 = plt.figure()
fig1.suptitle('Time Comparison')
ax1 = fig.add_subplot(111)
plt.plot(times, 'x', ms = 15)
ax1.set_xticklabels(names)
plt.show()
"""
fig1 = plt.figure()
fig1.suptitle('Algoritm vs Time')
ax1 = fig.add_subplot(111)
plt.plot(times, mean(results) 'x', ms = 15)
ax1.set_xticklabels(names)
plt.show()
"""
# Maybe come up with metric for best trade off between RMS and time
#%% At first it looks like linear regression gives us the best method in terms of trading off speed vs accuracy, however, it is already optimized
#is there a way to optimize RandomForests to get it to a low enough MSE that the trade off in time is worth it

rf = RandomForestRegressor(random_state = 42)
print(rf.get_params())

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
'''

k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring = 'neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    '''
    
