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
import sys, os
import matplotlib.pyplot as plt
import time
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
from sklearn.metrics import accuracy_score
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
import argparse

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tsfresh import extract_relevant_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from hpsklearn import HyperoptEstimator, gradient_boosting, xgboost_classification
from hyperopt import tpe,hp
from hyperopt.fmin import fmin
from xgboost import XGBClassifier
from datetime import datetime
import time
from hpsklearn import HyperoptEstimator, standard_scaler, xgboost_classification, random_forest, decision_tree, any_sparse_classifier, min_max_scaler
from hyperopt import tpe
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
print ('\n')

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

parser = argparse.ArgumentParser(description='Gets filenames for feature csvs')
parser.add_argument('-x', type=str, help='X feature dataset')
parser.add_argument('-y', type=str, help='Y feature dataset')
parser.add_argument('-cs','--constrainedScenarios', type=str, help='scenarios to only consider, comma seperated. eg. 0,1,2')
parser.add_argument('-cn','--constrainedNumDatasets', type=str, help='number of datasets from each scenario')
args = parser.parse_args()
#if extracted feature dataset is passed

def reduceScenarioData(scenarioCsvs,numOfScenarios=16,numDatasets=300):
	if args.constrainedScenarios is None and args.constrainedNumDatasets is None:
		return scenarioCsvs
	constrainedScenarios = list(map(int,args.constrainedScenarios.split(','))) if args.constrainedScenarios is not None else range(numOfScenarios)
	numDatasets = numDatasets if args.constrainedNumDatasets==None else args.constrainedNumDatasets
	sortedScenarios = {i:[] for i in range(numOfScenarios)}
	for i in scenarioCsvs:
		scenario = int(i.split('_')[1])
		if scenario in constrainedScenarios and len(sortedScenarios[scenario]) < numDatasets:
			sortedScenarios[scenario].append(i)
	finalDatasets = concatenate(list(sortedScenarios.values())).tolist()
	random.shuffle(finalDatasets)
	return finalDatasets

if args.x and args.y:
	print ('Importing datasets - x: {}, y: {}'.format(args.x, args.y))
	X_filtered = pandas.read_csv(args.x, index_col=0)
	y = pandas.read_csv(args.y, index_col=0)
	X_filtered = pandas.read_csv(args.x, header=0)
	X_filtered.astype({'id': int})
	print (X_filtered.info())
	y = pandas.read_csv(args.y, index_col=0, header=None)
	X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X_filtered, y, test_size=.4)
else:
	try:
		print ('Starting Data Handling')
		data_X = pandas.DataFrame([],columns=columnNames)
		data_Y = []
		outputDataset = reduceScenarioData(outputDataset)
		print_progress(0, len(outputDataset), prefix = 'Data Handling:')
		#Data Handling
		for dataSetID,data in enumerate(outputDataset):
			dataSetParams = data.replace('.csv','').split('_')
			dataSetParamsDict = {}
			#get dataset parameters as dict
			for idx,paramName in enumerate(["id","scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration", "ktSeverity", "vbusSeverity"]):
				dataSetParamsDict[paramName] = float(dataSetParams[idx])
			#dataset parameters as array excluding id num
			inputData = array([float(i) for i in dataSetParams[1:]])
			outputData = pandas.read_csv(outputFolder+"/"+data)
			embed()
			#ser input values to scenario numbers for target matrix
			#inputValues = [int(dataSetParamsDict[i]) for i in ['scenario']]
			inputValues = int(dataSetParamsDict['scenario'])

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
			preDataFrameResiduals = vstack([tile(dataSetID,datasetCutLength),arange(datasetCutLength),residuals]).T
			
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
			print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')

		data_Y = pandas.Series(data_Y)
	except Exception as err:
		print ("Data handling error:", err)
		embed()

	try:
		print ('\nStarting Feature Extraction')
		extractStartTime = time.time()
		df = data_X
		y = data_Y
		estim = HyperoptEstimator( classifier=any_sparse_classifier('clf'), 
	                            preprocessing=[min_max_scaler('min_max_scaler')],
	                            algo=tpe.suggest, trial_timeout=300)
		pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),('classifier', estim)])

		pipeline.set_params(augmenter__timeseries_container=data_X)
		pipeline.fit(data_X,data_Y)
		FCParameter = 'minimal'
		extraction_settings = {
			'comprehensive': ComprehensiveFCParameters(),
			'efficient': EfficientFCParameters(),
			'minimal': MinimalFCParameters(),
		}
		embed()
		X_filtered = extract_relevant_features(df, y, 
											   column_id='id', column_sort='time', 
											   default_fc_parameters=extraction_settings[FCParameter])
		
		#saving extracted features
		#https://github.com/zygmuntz/time-series-classification
		print (X_filtered.info())
		X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X_filtered, y, test_size=.4)
		saveTime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
		extractEndTime = time.time()
		#X_filtered.to_csv('X_{}_{}_{}.csv'.format(FCParameter,constainedScenariosFlag,saveTime))
		#y.to_csv('y_{}_{}_{}.csv'.format(FCParameter,constainedScenariosFlag,saveTime))
		#std = StandardScaler()
		#std.fit(X_filtered_train.values)
		#X_filtered_train = std.transform(X_filtered_train.values)
		#X_filtered_test = std.transform(X_filtered_test.values)
		print ('Feature Extraction Complete!!, it took {} seconds'.format(extractEndTime-extractStartTime))
	except Exception as err:
		print ("Feature Exraction Error:", err)
		embed()
#embed()
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
	'''
	print ('Starting ML Classifier')
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
	print ('best XGBClassifier:{}'.format(best))

	cl = XGBClassifier(**best)
	cl.fit(array(X_filtered_train), array(y_train))
	print cl.score(array(X_filtered_test), array(y_test))
	print classification_report(array(y_test), cl.predict(array(X_filtered_test))) 
	print ("The accuracy_score for XGBClassifier")
	print ("Training: {:6.5f}".format(accuracy_score(cl.predict(X_filtered_train), y_train)))
	print ("Test Set: {:6.5f}".format(accuracy_score(cl.predict(X_filtered_test), y_test)))

	'''
	'''
	hptrees = {
		'xgboost_classification': xgboost_classification('xgboost_classification'),
		'random_forest': random_forest('random_forest'),
		'decision_tree': decision_tree('decision_tree')
	}
	
	for hpmethodName, hpmethod in hptrees.items():
		print ('hpsklearn for', hpmethodName)
		estim = HyperoptEstimator( classifier=hpmethod, 
		                            preprocessing=[standard_scaler('standard_scaler')],
		                            algo=tpe.suggest, trial_timeout=300)
		
		estim.fit( X_filtered_train, y_train )
		
		print ('best model:', estim.best_model())
		print ('best score:', estim.score( X_filtered_test, y_test ))
	'''

	estim = HyperoptEstimator( classifier=any_sparse_classifier('clf'), 
	                            preprocessing=[min_max_scaler('min_max_scaler')],
	                            algo=tpe.suggest, trial_timeout=300)
	embed()
	#estim.fit( X_filtered_train, y_train.values)
	
	print ('best model:', estim.best_model())
	print ('best score:', estim.score( X_filtered_test.values, y_test.values))
	trees = {
		'RandomForestClassifier': RandomForestClassifier(),
		'DecisionTreeClassifier': DecisionTreeClassifier(), 
		'GradientBoostingClassifier': GradientBoostingClassifier(),
	}
	for mlName, tree in trees.items():
		tree.fit(X_filtered_train, y_train)
		print ("The accuracy_score for {}:".format(mlName))
		print ("Training: {:6.5f}".format(accuracy_score(tree.predict(X_filtered_train), y_train)))
		print ("Test Set: {:6.5f}".format(accuracy_score(tree.predict(X_filtered_test), y_test)))


except Exception as err:
	print ("ML Classifier Error:\n")
	exc_type, exc_obj, exc_tb = sys.exc_info()
	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	print (exc_type, fname, exc_tb.tb_lineno)
	embed()

embed()