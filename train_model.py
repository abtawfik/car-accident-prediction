######################
# Plotting libraries #
######################
#import matplotlib.pyplot as plt
#import seaborn as sns

###############
# Computation #
###############
import pandas as pd
import numpy as np
from toolz.curried import curry, compose
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.naive_bayes import CategoricalNB as CNB

######################
# Standard Libraries #
######################
from pathlib import Path

####################
# Internal package #
####################
from preproc import process_crash_report_data, encode_categorical_data
from preproc import split_train_and_test, Categorical, calc_mse
from preproc import read_crash_data
from benchmarks import AlwaysMean

#---------------------------------------
# Define the parameters
#---------------------------------------
input_file = Path('data/cpd-crash-incidents.csv')
features = ['Road_Conditions', 'Light_Condition', 'Weather']
target = 'Count'
time_column = 'Crash_Date'
crash_data = Categorical(input_file, features)
Xtrain, Ytrain = crash_data.Xtrain, crash_data.Ytrain
Xtest, Ytest   = crash_data.Xtest, crash_data.Ytest


#---------------------------------------
# Define a simple benchmark model
#---------------------------------------
always_mean = AlwaysMean().fit(crash_data.Xtrain, crash_data.Ytrain)
bench_pred  = always_mean.predict(crash_data.Xtest)
bench_error = calc_mse(crash_data.Ytest, bench_pred)


#-----------------------------------------------------
# Define a list of models to train as a first pass
#-----------------------------------------------------
names  = ['RandForestReg', 'RandForestClass', 'CatBayes']
models = [RFR(), RFC(), CNB()]
preds  = [m.fit(Xtrain, Ytrain).predict(Xtest) for m in models]
errors = {name : calc_mse(Ytest, pred) 
          for name,pred in zip(names,preds) }


#-----------------------------------------------------
# See if any of these models performed better than our benchmark
#-----------------------------------------------------
acceptable_models = {name : error for name,error in errors.items() 
                     if error < bench_error}
print(acceptable_models)
first_model = list(acceptable_models.keys())[0]
good_enough_model = dict(zip(names, models)).get(first_model)


#-----------------------------------------------------
# Save the model to be used for inference in the REST endpoint
#-----------------------------------------------------
import pickle
filename = 'acceptable_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(good_enough_model, f) 


#-----------------------------------------------------
# Save mapping metadata for use in the REST endpoint
#-----------------------------------------------------
filename = 'cat2code.pkl'
with open(filename, 'wb') as f:
    pickle.dump(crash_data.cat2code_map, f) 
    
filename = 'code2cat.pkl'
with open(filename, 'wb') as f:
    pickle.dump(crash_data.code2cat_map, f) 

