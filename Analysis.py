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
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV, train_test_split
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
#%% As with all data, we should do some basic exploritorty data analysis to see what we are dealing with.

data_train1.info()
data_train1.describe()
#I also like to see if I can broadly catch any 'NaN's in the dataframe
data_train1.isna().sum() # Finding NaNs

#Way to plot the blox plot of everything in one subgraph may add in later
'''
data_train1.plot(kind='box', subplots=True, layout=(5,6), sharex=False, sharey=False)
plt.show()
'''
#%%


plt.plot(data_train1['Unit'].value_counts(), 'o') # no of data points per unit
plt.show()
#%%
#It looks like sensor 22 through 26 is full of NaNs so in this case we can drop those columns. At the next iteration of this project we will
#have figure out a way to catch which columns have NaNs since in different data sets different sensors might be switched off or not recording
data_train1 = data_train1.drop(['sensor22','sensor23','sensor25','sensor26','sensor24'], axis=1)
#%%
#It looks like the operational settings are also pretty constant lets quickly plot them to also visually confirm
for setting in ['opset1', 'opset2', 'opset3']:
    plt.figure()
    plt.hist( data_train1[setting])
    plt.show()
#As one can see the setting is either constant or has a small distribution
#%% It also looks like this data set has multiple units.  For this inital EDA lets just look at Unit 1 and
    #what the sensor data looks lie for that 
unit_1_data = data_train1[data_train1['Unit'] == 1]
#%%Need To decide if we want to include the xlabel here
    
plt.figure(figsize=(20,40), facecolor='white')
plot_number = 1
for column in unit_1_data.columns:
    plt.subplot(5, 5, plot_number)
    plt.subplots_adjust(hspace = 0.5)
    plt.plot(unit_1_data['CycleNo'], unit_1_data[column])
    plt.title(column)
    #plt.xlabel('Cycle Number')
    plot_number += 1
plt.show()

#Cool we get a nice overview of what some of the sensor data looks like.  Some stay flat (maybe it's broken?), while others 
#shift with time.  One thing to point out is that these subplots are on different scales we will need to keep that
#in mind when we start trainning our models and potentially use standardization or normalization

#%%
#data_train1['RUL'] = ""


#Get a corrlation dataframe just to get a feel for what has the highest correlation
#data_train1.corr(method = 'pearson')
#df1 = data_train1.drop(['RUL'], axis=1)
'''
corr=df1.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');
'''
# print(max_cycle)
# print(max_cycle.loc[[1]])

# print(max_cycle.iloc[2, 1])

#%% Create the target data set
#Now we need to create a target variable.  In this case we want to be able to predict how many cycles are left
#before failure.  Thus, at each time cyclo number there should be a count down until the unit breaks
#We can find the max number of cycles per unit using a groupby function
max_cycle = data_train1.groupby('Unit')['CycleNo'].max().reset_index()
#I am a very visual person let's quickly plot this to see what the distribution looks like 
plt.figure()
plt.plot(max_cycle.iloc[:,0],max_cycle.iloc[:,1], 'x')
plt.show()
plt.figure()
plt.hist(max_cycle.iloc[:,1], bins = 10)
plt.title('Count of Max Cycle Numbers')
plt.ylabel('Max Cycle Number')
plt.show()
#So it looks like we have a decent distribution of cycle numbers
#%%
#Now we need to correspond each 'cycleno' entry to its unit an that units max cycle number.  Subtracting the two 
#will give us the number of cycles left until the part breaks

#Note here we use a for loop this seems like the best option although we welcome any suggestions for lowering the 
#time complexitiy 
#Another note using .iloc one has to subtract 1 from x because the index starts at zero.  Took more time than we 
#care to admit to remember that
for x in max_cycle.iloc[:, 0]:
    count_cycle = max_cycle.iloc[x-1, 1]
    data_train1.loc[data_train1['Unit'] == x, 'RUL'] = count_cycle - data_train1['CycleNo']

#Now we have a compete feature set with a corresponding target variable
    
#%%Going back to our EDA we want to normalize or standardize our data since eat sensor
    #has different scales.  Here we pick normalization as it will remove the effect of outliers
    #Additionally, standardization assumes the data is distriubted normaly which we can't guarantee
    
df = data_train1.drop(['RUL','Unit','CycleNo'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.columns = df.columns

#%% Now we want to iterate through some different models quickly to get a sense of what would work best.
#additionally we want to keep track of how long each model takes to fit.  Ideally we would get the most accurate model
#with the fastest time as that would allow us to best us this in real life.  

#Toggle this on and off for warnings when we write it up can say: Let this on when developing but turned it off cause SVR was giving 
#us "Future warnings" which have no effect on the quick look 
warnings.filterwarnings("ignore")

num_folds = 7
seed = 5
 
X_train, X_test, y_train, y_test = train_test_split(df_normalized, data_train1["RUL"], test_size=0.3, random_state=5)

#%% Let's Now test some models to see most of these are simple models.  We threw in an ensemble method just to see how that family of methods would fair.
def compare_alorithms(alorithms, X_data, y_data, scoring = 'neg_mean_squared_error', kfold = 3):
    '''
    In take a list of names: alorithms to compare run time
    '''
    results = []
    names = []
    times = []
    results_mean =[]
    results_std = []
    for name, model in alorithms:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        start = time.time()
        cv_results = cross_val_score(model, X_data, y_data, cv=kfold, scoring='neg_mean_squared_error')
        results.append(cv_results)
        results_mean.append(cv_results.mean())
        results_std.append(cv_results.std())
        run_time = time.time()-start
        names.append(name)
        times.append(run_time)
        ##print(message)
        
    alorithms_df = pd.DataFrame(
        {'Names': names,
         'Mean MSE': results_mean,
         'Std': results_std,
         'Time (s)': times,
        })
    return alorithms_df, results


models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('Bagging', RandomForestRegressor()))

single_model_compare, results = compare_alorithms(models, X_train, y_train, kfold = 7)

#%% Create Visual for people   #Come back and fix outputs need to include results in the output for boxplots 

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(single_model_compare['Names'])
plt.show()    
# Add a plot of MSE vs Time labeled with different regression algorithums when we come back 
fig1 = plt.figure()
fig1.suptitle('Time Comparison')
ax1 = fig.add_subplot(111)
plt.plot(single_model_compare['Time(s)'], 'x', ms = 15)
ax1.set_xticklabels(single_model_compare['Names'])
plt.show()

print(single_model_compare)
# Maybe come up with metric for best trade off between RMS and time
#%% It looks like the ensemble method of random forest might be the best option.  Let's use the same iterative method to get a quick idea of 
#how other ensemble methods would fair.  Let's leave the Random forest Regressor in there so we can easily plot it to see how it matched up 
ensemble_methods = []
ensemble_methods.append(('AdaBoost', AdaBoostRegressor()))
ensemble_methods.append(('GradientBoosting', GradientBoostingRegressor()))
ensemble_methods.append(('ExtraTree', ExtraTreesRegressor()))
ensemble_methods.append(('Random Forest', RandomForestRegressor()))

ensemble_methods_df, results = compare_alorithms(ensemble_methods, X_train, y_train, kfold = 7)

#%%Let's make the same sort of plots to visually show which model does the best
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensemble_methods_df['Names'])
plt.show()    
# Add a plot of MSE vs Time labeled with different regression algorithums when we come back 
fig1 = plt.figure()
fig1.suptitle('Time Comparison')
ax1 = fig.add_subplot(111)
plt.plot(ensemble_methods_df['Time(s)'], 'x', ms = 15)
ax1.set_xticklabels(ensemble_methods_df['Names'])
plt.show()

print(ensemble_methods_df)


#%% It looks like gradient boosting is the best method and is rather quick.  One side note it is that both Random forest and Extra tree's max
#accuracy seem to lay within Gradient boosting's standard deviations.  We'll come back and look to see what happens if we optimize one or both of those
#methods.  Another note, gradient boosting seems to be sensative to outliers and noise so might not be great for this method but we've normalized the 
#data set does that matter?

GBM = GradientBoostingRegressor(random_state = 42)


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start = 3, stop = 50, num = 10)]
#post says this should be ~1% total samples but default is 2 so trying a range
min_samples_split = [2, 10, 100] 
#Want to pick small values to prevent over fitting 
min_samples_leaf = [1, 20, 40, 60, 80]

learning_rate = [0.002, 0.02, 0.2]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}



GBM_random = RandomizedSearchCV(estimator =GBM, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=42, n_jobs = -1)
GBM_random.fit(X_train, y_train)
print(GBM_random.best_params_)
estimators = {'n_estimators': 40, 'min_samples_split': 100, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'max_depth': 5, 'learning_rate': 0.2}
estimators = {'n_estimators': 60, 'min_samples_split': 100, 'min_samples_leaf': 60, 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.2}

#%%
#Great now let's see how much this hyperparamter tuning helped our accuracy by testing it vs the base model 

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    RMSE = (mean_squared_error(test_labels, predictions))**0.5
    print('Model Performance')
    print('Root Mean Squared Error: {:0.4f} cycles.'.format(np.mean(RMSE)))

base_model = GradientBoostingRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = GradientBoostingRegressor(**estimators)
best_random.fit(X_train, y_train)
random_accuracy = evaluate(best_random, X_test, y_test)

#Cool we've been able to get down to about 7 cycles less than then the baseline model 

#%% Let's do a more focused gridsearchCV to see if we can get the model further optimized
param_grid = {'n_estimators': [ 55,60,65,70],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [2,3,4,5],
               'min_samples_split': [95,100,105, 110],
               'min_samples_leaf': [55,60,65],
               'learning_rate': [0.15,0.2,0.25]}
# Create a based model
GBM = GradientBoostingRegressor()
# Instantiate the grid search model
GBM_grid_search = GridSearchCV(estimator = GBM, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
GBM_grid_search.fit(X_train, y_train)
print(GBM_grid_search.best_params_)
#%%
#Now that we have refined our parameters a bit more let's if that improves our accuracy on the test data

estimators = {'learning_rate': 0.15, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 60, 'min_samples_split': 110, 'n_estimators': 60}
best_grid = GradientBoostingRegressor(**estimators)
best_grid.fit(X_train, y_train)
grid_accuracy = evaluate(best_random, X_test, y_test)

#Dang looks like we have reached a semi limit on optimization for this model.
##need to write a bit more 
#%% As noted another model to be used is the random forest which at the upper part of its broad RMSE range is in the range of the Gradient booster
#Thus, let's see if we can optimize it in the same fashion and improve upon our gradient boosting model

#Dont need to run this every time I have put the parameters into a dictionary in the next section for model performance 

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
#These come from doing a big random search in the section above
params = {'n_estimators': 890, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': True}
params = {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': False}


base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = RandomForestRegressor(**params)
best_random.fit(X_train, y_train)
random_accuracy = evaluate(best_random, X_test, y_test)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_accuracy = evaluate(linear_model, X_test, y_test)
#%%Now do gridsearch around the best parameters we found 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40],
    'max_features': ['sqrt', 'auto'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [400, 600, 800, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
grid_search.best_params_

#%%Okay it seems like we get a slight inprovement in the decemical places for the Random forest 
#I believe if this was a real life problem I would suggeset using the gradient boosting model since it is faster to train and use in real time.
#This can be seen in the section below where I simply time both fitting.
#again this is going back to that trade off in time vs accuracy.  This could also shift based on the equipment.  If it's better to take the time to more accruate by
#a few cycles maybe across 1000s of fans that would outweigh being able to predict more in real time.  
#One final note, gradient boosting tends to be sensative to outliers so that would have to be investigated further 

start = time.time()
params = {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': False}
best_random = RandomForestRegressor(**params)
best_random.fit(X_train, y_train)
run_time = time.time()-start
print(run_time)

start = time.time()
estimators = {'learning_rate': 0.15, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 60, 'min_samples_split': 110, 'n_estimators': 60}
best_grid = GradientBoostingRegressor(**estimators)
best_grid.fit(X_train, y_train)
run_time = time.time()-start
print(run_time)



