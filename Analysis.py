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
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import seaborn as sns; sns.set()
import time
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
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

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]

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

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
#%%
#Compare this to the base model 
#Issue here is that we have zeros in our test labels which is causing accuracy to go go to inf% 
'''
Orignal Function pulled from article
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #print(predictions)
    print(test_labels.describe())
    errors = abs(predictions - test_labels)
    #print(errors)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} cycles.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
'''
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    #errors = abs(predictions - test_labels)
    RMSE = (mean_squared_error(test_labels, predictions))**0.5
    print(RMSE)
    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f} cycles.'.format(np.mean(RMSE)))

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_test, y_test)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)
