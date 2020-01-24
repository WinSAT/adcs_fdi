##################################################
## Fault isolation of reaction wheels onboard 3-axis controlled in-orbit satellite 
## using ensemble machine learning techniques
##################################################
## MIT License
## Copyright (c) 2019 Atilla Saadat
##################################################
## Author: Afshin Rahimi, Atilla Saadat
## Version: 0.3
## Email: afshin.rahimi@uwindsor.ca, atilla.saadat@uwindsor.ca
## Status: In Development
##################################################

from numpy import *
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython import embed
#import tsfresh

from scipy.stats import skew
from scipy.stats import kurtosis
#from statsmodels import robust
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
#from tsfresh import extract_relevant_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from hpsklearn import HyperoptEstimator, gradient_boosting
from sklearn import svm
#from hyperopt import base
#from hyperopt import hp, tpe
#from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
#from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.pipeline import Pipeline


from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#import xgboost as xgb
#
#import lightgbm as lgbm

outputFolder = "output_300_constSeverity_csvs"
stepsizeFreq = 10

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

#cut the dataset for testing with smaller total datasets
'''
listCutFactor = 1
if listCutFactor > 1:
	outputDataset = outputDataset[::listCutFactor]
'''
#sets headers for faulty and nominal time series
xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
xNamesTrain = ['id', 'time'] + ['_'.join(i.split('_')[:-1]) for i in xNamesFaulty]

print ("started data handling")
startTime = time.time()

allDataX = {}
allDataY = []
resultsList = []
inputValuesList = []
maxTimeSequence = 0
for dataSetId,data in enumerate(outputDataset):
	dataSetParams = data.replace('.csv','').split('_')
	dataSetParamsDict = {}
	for idx,paramName in enumerate(["id","scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration", "ktSeverity", "vbusSeverity"]):
		dataSetParamsDict[paramName] = float(dataSetParams[idx])
 	#if scenarioNum > 4: # only consider specific scenario numbers
 	#	continue
	inputData = array([float(i) for i in dataSetParams[1:]])
	outputData = pd.read_csv(outputFolder+"/"+data)
	#ser input values to scenario numbers for target matrix
	inputValues = [int(dataSetParamsDict[i]) for i in ['scenario']]
	if dataSetParamsDict['ktDuration'] != 0.0:
		outputData = outputData[int(dataSetParamsDict['ktInception'])*stepsizeFreq:int(dataSetParamsDict['ktInception']+dataSetParamsDict['ktDuration'])*stepsizeFreq+1]
	#normalized timeseries
	faulty = (outputData[xNamesFaulty].values).T
	nominal =  (outputData[xNamesNominal].values).T

	#filter implementation (unused)
	#for i in range(len(faulty)):
	#	faulty[i] = savgol_filter(faulty[i],41,3)
	#	nominal[i] = savgol_filter(nominal[i],41,3)

	residuals = faulty - nominal
	
	'''
	#plot datset for inspection
	if inputValues[0] != 0:
		fig, ax = plt.subplots(3,2)
		fig.suptitle('q residuals')
		[ax[i][0].plot(nominal[i],'b') for i in range(3)]
		[ax[i][0].plot(faulty[i],'r') for i in range(3)]
		[ax[i][1].plot(residuals[i]) for i in range(3)]
		fig, ax = plt.subplots(3,2)
		fig.suptitle('omega residuals')
		[ax[i-3][0].plot(nominal[i],'b') for i in range(3)]
		[ax[i-3][0].plot(faulty[i],'r') for i in range(3)]
		[ax[i-3][1].plot(residuals[i]) for i in range(3,6)]
		#embed()
	'''

	#nominal = StandardScaler().fit_transform(nominal)
	#faulty = StandardScaler().fit_transform(faulty)
	#outputValues = [np.sqrt(mean_squared_error(nominal.T[i],faulty.T[i])) for i in range(len(xNames))] 	#.flatten()
	#resultsList['nominal'] = [nominal, array([0])] #hard coded nominal output
	#if inputValues[0] != 0:
	if maxTimeSequence < residuals.shape[1]:
		maxTimeSequence = residuals.shape[1]
	resultsList.append(residuals.T)
	inputValuesList.append(inputValues)
	#data handling
inputValuesList = array(inputValuesList).flatten()
try:
	new_seq = []
	for one_seq in resultsList:
	    len_one_seq = len(one_seq)
	    last_val = one_seq[-1]
	    n = maxTimeSequence - len_one_seq
	   
	    to_concat = repeat(one_seq[-1], n).reshape(one_seq.shape[1], n).transpose()
	    new_one_seq = concatenate([one_seq, to_concat])
	    new_seq.append(new_one_seq)
	final_seq = stack(new_seq)
except Exception as e:
	print ('padding fail:', e)
	embed()

X_train, X_test, y_train, y_test = train_test_split(final_seq, inputValuesList, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
#TODO: Truncate to 90th quartile, as shown in pd.Series(resultsList).describe()
'''
#truncate the sequence to length 60
from keras.preprocessing import sequence
seq_len = 60
final_seq=sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')
'''
print ('starting model creation')
try:
	model = Sequential()
	model.add(LSTM(256, input_shape=(maxTimeSequence, 6)))
	model.add(Dense(1, activation='sigmoid'))
	
	print(model.summary())
	
	adam = Adam(lr=0.001)
	chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.fit(X_train, y_train, epochs=200, batch_size=128, callbacks=[chk], validation_data=(X_val,y_val))
	
	#loading the model and checking accuracy on the test data
	model = load_model('best_model.pkl')
	
	from sklearn.metrics import accuracy_score
	test_preds = model.predict_classes(X_test)
	print(accuracy_score(y_test, test_preds))
except Exception as e:
	print ('model failed:', e)
	embed()