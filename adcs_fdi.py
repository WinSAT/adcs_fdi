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
import pandas
import time
import sys

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		bar_length  - Optional  : character length of bar (Int)
	"""
	str_format = "{0:." + str(decimals) + "f}"
	percents = str_format.format(100 * (iteration / float(total)))
	filled_length = int(round(bar_length * iteration / float(total)))
	bar = '*' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

#import xgboost as xgb
#
#import lightgbm as lgbm

outputFolder = "output_300_constSeverity_csvs"
stepsizeFreq = 10.0

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
columnNames = ['id','time']+[i.split('_')[0]+'_e' for i in xNamesFaulty]

data_X = pandas.DataFrame([],columns=columnNames)
data_Y = []
print_progress(0, len(outputDataset), prefix = 'Data Handling:')
try:
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
		#inputValues = [int(dataSetParamsDict[i]) for i in ['scenario']]
		inputValues = dataSetParamsDict['scenario']
		if dataSetParamsDict['ktDuration'] != 0.0:
			outputData = outputData[int(dataSetParamsDict['ktInception']*stepsizeFreq):int((dataSetParamsDict['ktInception']+dataSetParamsDict['ktDuration'])*stepsizeFreq+1)]
		#normalized timeseries
		faulty = normalize((outputData[xNamesFaulty].values).T,axis=0)
		nominal =  normalize((outputData[xNamesNominal].values).T,axis=0)
		datasetCutLength = faulty.shape[1]
	
		#filter implementation (unused)
		#for i in range(len(faulty)):
		#	faulty[i] = savgol_filter(faulty[i],41,3)
		#	nominal[i] = savgol_filter(nominal[i],41,3)
	
		residuals = faulty - nominal
		preDataFrameResiduals = vstack([tile(dataSetId,datasetCutLength),arange(datasetCutLength),residuals]).T
		
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
		data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrameResiduals,columns=columnNames)],ignore_index=True)
		data_Y.append(inputValues)
		#data handling
		print_progress(dataSetId, len(outputDataset), prefix = 'Data Handling:')

	data_Y = pandas.Series(data_Y)
except Exception as err:
	print "Data handling error:", err
	embed()

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tsfresh import extract_relevant_features
from sklearn.ensemble import GradientBoostingClassifier

from hpsklearn import HyperoptEstimator, gradient_boosting, xgboost_classification
from hyperopt import tpe,hp
from hyperopt.fmin import fmin
from xgboost import XGBClassifier

try:
	df = data_X
	y = data_Y
	extraction_settings = ComprehensiveFCParameters()
	#extraction_settings = EfficientFCParameters()
	#extraction_settings = MinimalFCParameters()
	
	X_filtered = extract_relevant_features(df, y, 
										   column_id='id', column_sort='time', 
										   default_fc_parameters=extraction_settings)
	
	#saving extracted features
	#https://github.com/zygmuntz/time-series-classification
	X_filtered.to_csv('X_filtered.csv')
	
	print X_filtered.info()
	X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X_filtered, y, test_size=.4)

except Exception as err:
	print "Feature Exraction Error:", e
	embed()
embed()
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

gini_scorer = make_scorer(gini_xgb, greater_is_better=True, needs_proba=True)

try:

	def objective(params,X=X_filtered, Y=y):
		#params = {
		#	'max_depth': int(params['max_depth']),
		#	'gamma': "{:.3f}".format(params['gamma']),
		#	'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
		#}
		
		clf = XGBClassifier(
			#n_estimators=250,
			#learning_rate=0.05,
			#n_jobs=4,
			**params
		)
		
		score = cross_val_score(clf, X, Y, cv=StratifiedKFold()).mean()
		print("Gini {:.3f} params {}".format(score, params))
		return score

	max_leaf_n = [-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	space = {
		'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
		'gamma': hp.uniform('gamma', 0.0, 0.5),
		'learning_rate': hp.choice('learning_rate', array(range(1, 10))/100.),
		'n_estimators': hp.choice('n_estimators', array(range(100, 10000, 100))),
		'subsample': hp.choice('subsample', array(range(5, 10))/10.),
		'max_depth': hp.choice('max_depth', array(range(3, 11))),
		'max_features': hp.choice('max_features', array(range(5, 10))/10.),
		'max_leaf_nodes':hp.choice('max_leaf_nodes', array(max_leaf_n))
	}

	best = fmin(fn=objective,
				space=space,
				algo=tpe.suggest,
				max_evals=10)
	print 'best XGBClassifier:{}'.format(best)

	cl = XGBClassifier(**best)
	cl.fit(array(X_filtered_train), array(y_train))
	print cl.score(array(X_filtered_test), array(y_test))
	print classification_report(array(y_test), cl.predict(array(X_filtered_test))) 

except Exception as err:
	print "ML Classifier Error:", err
	embed()

embed()