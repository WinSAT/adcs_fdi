##################################################
## Fault isolation of reaction wheels onboard 3-axis controlled in-orbit satellite 
## using ensemble machine learning techniques
##################################################
## MIT License
## Copyright (c) 2019 Atilla Saadat
##################################################
## Author: Afshin Rahimi, Atilla Saadat
## Version: 0.3
## Mmaintainer: Atilla Saadat
## Email: Afshin.Rahimi@uwindsor.ca, Atilla.Saadat@uwindsor.ca
## Status: In Development
##################################################

from numpy import *
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython import embed
import tsfresh

from scipy.stats import skew
from scipy.stats import kurtosis
from statsmodels import robust
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from tsfresh import extract_relevant_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from hpsklearn import HyperoptEstimator, gradient_boosting
from sklearn import svm
from hyperopt import base
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.pipeline import Pipeline

import xgboost as xgb

import lightgbm as lgbm

outputFolder = "output_300_constSeverity_csvs"

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

#cut the dataset for testing with smaller total datasets
listCutFactor = 1
if listCutFactor > 1:
	outputDataset = outputDataset[::listCutFactor]

timeframe = 10
csvTimestep = 0.1

inputDict = {}
outputDict = {}

#sets headers for faulty and nominal time series
xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
xNamesTrain = ['id', 'time'] + ['_'.join(i.split('_')[:-1]) for i in xNamesFaulty]

xValues = []
yValues = []

print ("started data handling")
startTime = time.time()

'''
#manually calculate timeseries features (unused after tsfresh implementation)
def generateTimeseriesFeatures(data):
	results = []
	#outputValuesQ.append(np.sqrt(mean_squared_error(nom,fal)))
	results.append(max(data))
	results.append(std(data))
	results.append(min(data))
	results.append(mean(data))
	results.append(skew(data))
	results.append(kurtosis(data))
	results.append(robust.mad(data))
	#outputValuesQ.append(np.max(residual)/np.sqrt(mean_squared_error(nom,fal)))
	#outputValuesQ.append(np.sqrt(mean_squared_error(nom,fal))/np.mean(residual))
	#outputValuesQ.append(np.max(residual)/np.mean(residual))
	#outputValuesQ.append(mean_absolute_error(nom,fal))
	return results
'''

allDataX = {}
allDataY = []
idCounter = -1
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
	inputValues = array([int(dataSetParamsDict[i]) for i in ['scenario']])
	embed()
	if dataSetParamsDict['ktDuration'] != 0.0:
		outputData = outputData[int(dataSetParamsDict['ktInception'])*10:int(dataSetParamsDict['ktInception']+dataSetParamsDict['ktDuration'])*10+1]
	#normalized timeseries
	faulty = normalize(outputData[xNamesFaulty].values).T
	nominal =  normalize(outputData[xNamesNominal].values).T

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
	resultsList = {}
	#resultsList['nominal'] = [nominal, array([0])] #hard coded nominal output
	#if inputValues[0] != 0:
	resultsList['residuals'] = [residuals, inputValues]
	trainSetX = []
	trainSetY = []
	#data handling
	for dataType, currentDataset in resultsList.items():
		idCounter += 1
		currentTrainSetX = currentDataset[0]
		currentTrainSetY = currentDataset[1]
		#creating ID and TIME entries to the dataset
		scenarioData = (vstack((tile([idCounter],len(currentTrainSetX.T)),arange(len(currentTrainSetX.T)),currentTrainSetX))).tolist()
		#embed()
		#appending required timeseries daata
		for col in xNamesTrain:
			allDataX.setdefault(col,[]).append(scenarioData[xNamesTrain.index(col)])
		allDataY.append(currentTrainSetY[0])
		#if addIdxNum == 1:
		#	embed()
		#xValues.append(array([generateTimeseriesFeatures(ts) for ts in currentTrainSetX]).ravel())
		#yValues.append(currentTrainSetY)
	#yValues = inputValues if yValues is None else np.vstack((yValues,inputValues))
	#xValues = outputValues if xValues is None else np.vstack((xValues,outputValues))

#create pd objects as stated in tsfresh docs
allDataX = pd.DataFrame({k:array(val).flatten() for k,val in allDataX.items()})
allDataY = pd.Series(allDataY)
print 'done DataHandling'

#feature extraction
#extraction_settings = ComprehensiveFCParameters()
#extraction_settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
extraction_settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
features_filtered_direct = extract_relevant_features(allDataX, allDataY,column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
base.have_bson = False
# Load Data
# ...

#scoring test based on online resource
def gini(truth, predictions):
    g = asarray(c_[truth, predictions, arange(len(truth)) ], dtype=float)
    g = g[lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)

def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)

#train test spliting at 20%
X_train, X_test, y_train, y_test = train_test_split(features_filtered_direct, allDataY, test_size=.2)

#hyperparameter optimization and fitting for sklearn randomforest
try:
	def objective(params,X=features_filtered_direct, Y=allDataY):
	    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}
	    #params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}
	    #clf = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
        #    ('classifier', RandomForestClassifier(n_jobs=4, class_weight='balanced', **params))])
	    #clf.set_params(augmenter__timeseries_container=allDataX)
	    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)
	    #clf = GradientBoostingClassifier(learning_rate=0.05, **params)
	    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
	    print("Gini {:.3f} params {}".format(score, params))
	    return score
	
	space = {
	    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
	    'max_depth': hp.quniform('max_depth', 1, 10, 1)
	}
	
	best = fmin(fn=objective,
	            space=space,
	            algo=tpe.suggest,
	            max_evals=10)
	print("SKlearn Hyperopt estimated optimum {}".format(best))
	best['n_estimators'] = int(best['n_estimators'])
	best['max_depth'] = int(best['max_depth'])
	#cl = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
	#	('classifier', RandomForestClassifier(n_jobs=4, class_weight='balanced', **best))])
	#cl.set_params(augmenter__timeseries_container=df_ts)
	cl = RandomForestClassifier(**best)
	#cl = GradientBoostingClassifier(**best)
	cl.fit(X_train, y_train)
	print cl.score(X_test, y_test)
	print classification_report(y_test, cl.predict(X_test)) 
except Exception as e: 
	print(e)
	embed()
#hyperparameter optimization and fitting for xgb classifier
try:
	def objective(params,X=features_filtered_direct, Y=allDataY):
	    params = {
	        'max_depth': int(params['max_depth']),
	        'gamma': "{:.3f}".format(params['gamma']),
	        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
	    }
	    
	    clf = xgb.XGBClassifier(
	        n_estimators=250,
	        learning_rate=0.05,
	        n_jobs=4,
	        **params
	    )
	    
	    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
	    print("Gini {:.3f} params {}".format(score, params))
	    return score

	space = {
	    'max_depth': hp.quniform('max_depth', 2, 8, 1),
	    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
	    'gamma': hp.uniform('gamma', 0.0, 0.5),
	}

	best = fmin(fn=objective,
	            space=space,
	            algo=tpe.suggest,
	            max_evals=10)
	best['max_depth'] = int(best['max_depth'])
	print("XGB Hyperopt estimated optimum {}".format(best))

	cl = xgb.XGBClassifier(**best)
	cl.fit(array(X_train), array(y_train))
	print cl.score(array(X_test), array(y_test))
	print classification_report(array(y_test), cl.predict(array(X_test))) 
except Exception as e: 
	print(e)
	embed()

#hyperparameter optimization and fitting for lgbm classifier
try:
	def objective(params,X=features_filtered_direct, Y=allDataY):
	    params = {
	        'num_leaves': int(params['num_leaves']),
	        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
	    }
	    
	    clf = lgbm.LGBMClassifier(
	        n_estimators=500,
	        learning_rate=0.01,
	        **params
	    )
	    
	    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
	    print("Gini {:.3f} params {}".format(score, params))
	    return score
	
	space = {
	    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
	    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
	}
	
	best = fmin(fn=objective,
	            space=space,
	            algo=tpe.suggest,
	            max_evals=10)
	best['num_leaves'] = int(best['num_leaves'])
	print("LGBM Hyperopt estimated optimum {}".format(best))
	cl = lgbm.LGBMClassifier(**best)
	cl.fit(array(X_train), array(y_train))
	print cl.score(array(X_test), array(y_test))
	print classification_report(array(y_test), cl.predict(array(X_test))) 
except Exception as e: 
	print(e)
	embed()

#cl = HyperoptEstimator(classifier=gradient_boosting('myGB'))

embed()
'''

dataset = {'id': tile(currentTrainSetY,len(currentTrainSetX.T)), 'time': arange(len(currentTrainSetX.T))}
dataset.update({})
print "finished data handling: {:.4}s".format(time.time()-startTime)
yValues = ravel(yValues)
xValues = array(xValues)

from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.3, random_state=123) #explain hypervariables
print "\n"
clfs = {
	"AdaBoostClassifier DecisionTreeClassifier": AdaBoostClassifier(DecisionTreeClassifier()),
	"AdaBoostClassifier RandomForestClassifier": AdaBoostClassifier(RandomForestClassifier(n_estimators=200,random_state=0)),
	"MLP": MLPClassifier(),
	"kNN": KNeighborsClassifier(n_neighbors=3)
	#"AdaBoostClassifier RandomForestClassifier": MultiOutputClassifier(AdaBoostClassifier(RandomForestClassifier(n_estimators=200, max_depth=5, max_features="auto",random_state=0))),
	}
for clfName, clf in clfs.items():
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	print "{} accuracy_score: {}".format(clfName,accuracy_score(y_pred,y_test))
	#from IPython import embed; embed()
	if clfName == "AdaBoostClassifier RandomForestClassifier":
		print "Num Datapoints:", len(y_test)
		cm = confusion_matrix(y_test, y_pred, labels=range(5))
		cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
		print cm
		rfpred = y_pred

embed()

'''