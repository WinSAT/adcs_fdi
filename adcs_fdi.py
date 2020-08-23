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
from sklearn.metrics import confusion_matrix

from hyperopt import tpe,hp
from hyperopt.fmin import fmin
#from xgboost import XGBClassifier
from datetime import datetime
import time
from hpsklearn import HyperoptEstimator, standard_scaler, xgboost_classification, random_forest, decision_tree, any_classifier, min_max_scaler, gradient_boosting, any_preprocessing
from hyperopt import tpe
from sklearn.pipeline import Pipeline
from tsfresh.examples import load_robot_execution_failures
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import RelevantFeatureAugmenter
import pickle

print ('\n')

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=50):
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

outputFolder = "output_625"
stepsizeFreq = 10.0

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
columnNames = ['id','time']+[i.split('_')[0] for i in xNamesFaulty]

parser = argparse.ArgumentParser(description='Gets filenames for feature csvs')
parser.add_argument('--x', type=str, help='X pandas dataset')
parser.add_argument('--y', type=str, help='Y pandas dataset')
parser.add_argument('--fx', type=str, help='X features dataset')
parser.add_argument('--fy', type=str, help='Y features dataset')
parser.add_argument('-cs','--constrainedScenarios', type=str, help='scenarios to only consider, comma seperated. eg. 0,1,2')
parser.add_argument('-cn','--constrainedNumDatasets', type=str, help='number of datasets from each scenario')
args = parser.parse_args()
#if extracted feature dataset is passed

def reduceScenarioData(scenarioCsvs,numOfScenarios=16,numDatasets=300):
	if args.constrainedScenarios is None and args.constrainedNumDatasets is None:
		return scenarioCsvs
	constrainedScenarios = list(map(int,args.constrainedScenarios.split(','))) if args.constrainedScenarios is not None else range(numOfScenarios)
	numDatasets = int(numDatasets) if args.constrainedNumDatasets==None else int(args.constrainedNumDatasets)
	sortedScenarios = {i:[] for i in range(numOfScenarios)}
	for i in scenarioCsvs:
		scenario = int(i.split('_')[1])
		if scenario in constrainedScenarios and len(sortedScenarios[scenario]) < numDatasets:
			sortedScenarios[scenario].append(i)
	finalDatasets = concatenate(list(sortedScenarios.values())).tolist()
	random.shuffle(finalDatasets)
	return finalDatasets

feature_settings = {
	"abs_energy" : None,
	"absolute_sum_of_changes" : None,
	"approximate_entropy": [{"m": 2, "r": r} for r in [.1, .3, .5, .7, .9]],
	"agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
	"binned_entropy" : [{"max_bins" : 10}],
	"c3": [{"lag": 3}],
	"cid_ce": [{"normalize":False}],
	"count_above_mean" : None,
	"count_below_mean" : None,
	"energy_ratio_by_chunks": [{"num_segments" : 5, "segment_focus": i} for i in range(5)],
	"first_location_of_maximum" : None,
	"first_location_of_minimum" : None,
	"has_duplicate" : None,
	"has_duplicate_max" : None,
	"has_duplicate_min" : None,
	"index_mass_quantile": [{"q": q} for q in [0.25, 0.5, 0.75]],
	"kurtosis" : None,
	"last_location_of_maximum" : None,
	"last_location_of_minimum" : None,
	"length" : None,
	"longest_strike_above_mean" : None,
	"longest_strike_below_mean" : None,
	"maximum" : None,
	"max_langevin_fixed_point": [{"m": 3, "r": 30}],
	"mean" : None,
	"mean_abs_change" : None,
	"mean_change" : None,
	"median" : None,
	"minimum" : None,
	"number_peaks" : [{"n" : 3}],
	"number_crossing_m" : [{"m" : 0}],
	"percentage_of_reoccurring_values_to_all_values" : None,
	"percentage_of_reoccurring_datapoints_to_all_datapoints" : None,
	"quantile": [{"q": q} for q in [.1, .2, .3, .4, .6, .7, .8, .9]],
	"range_count": [{"min": -1e12, "max": 0}, {"min": 0, "max": 1e12}],
	"ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3]],
	"sample_entropy" : None,
	"skewness" : None,
	"standard_deviation" : None,
	"sum_of_reoccurring_data_points" : None,
	"sum_of_reoccurring_values" : None,
	"sum_values" : None,
	"symmetry_looking": [{"r": r * 0.25} for r in range(4)],
	"variance" : None,
	"value_count" : [{"value" : 0}],
	"variance_larger_than_standard_deviation" : None,
}

if args.x and args.y:
	print ('Importing datasets - x: {}, y: {}'.format(args.x, args.y))
	#X_filtered = pandas.read_csv(args.x, index_col=0)
	data_Y = pandas.read_csv(args.y, index_col=0,squeeze=True) #pandas series
	data_X = pandas.read_csv(args.x, index_col=0)
	data_X = data_X.astype({'id': int})
	#print (X_filtered.info())
	#y = pandas.read_csv(args.y, index_col=0, header=None)
elif not args.x and not args.y and not args.fx and not args.fy:
	try:
		print ('Starting Data Handling')
		data_X = pandas.DataFrame([],columns=columnNames)
		data_Y = []
		#outputDataset = reduceScenarioData(outputDataset)
		print_progress(0, len(outputDataset), prefix = 'Data Handling:')
		#Data Handling
		dataSetIdCounter = 0
		for dataSetID,data in enumerate(outputDataset):
			dataSetParams = data.replace('.csv','').split('_')
			if int(dataSetParams[1]) not in [0,1,2,3,4]:
				print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')
				continue
			dataSetParamsDict = {}
			#get dataset parameters as dict
			for idx,paramName in enumerate(["id","scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration", "ktSeverity", "vbusSeverity"]):
				dataSetParamsDict[paramName] = float(dataSetParams[idx])
			#dataset parameters as array excluding id num
			inputData = array([float(i) for i in dataSetParams[1:]])
			outputData = pandas.read_csv(outputFolder+"/"+data)
			#ser input values to scenario numbers for target matrix
			#inputValues = [int(dataSetParamsDict[i]) for i in ['scenario']]
			#inputValues = int(dataSetParamsDict['scenario'])
			inputValues = array([int(dataSetParamsDict[i]) for i in ["scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration"]])

			#if dataSetParamsDict['ktDuration'] != 0.0:
			#   outputData = outputData[int(dataSetParamsDict['ktInception']*stepsizeFreq):int((dataSetParamsDict['ktInception']+dataSetParamsDict['ktDuration'])*stepsizeFreq+1)]
			#normalized timeseries
			#faulty = normalize((outputData[xNamesFaulty].values).T,axis=0)
			faulty = (outputData[xNamesFaulty].values).T
			#nominal =  normalize((outputData[xNamesNominal].values).T,axis=0)
			nominal =  (outputData[xNamesNominal].values).T
			'''
			nominal =  normalize(outputData[xNamesNominal].values,axis=0)
			preDataFrame = vstack([tile(dataSetIdCounter,nominal.shape[0]),array(outputData['time']),nominal.T]).T
			data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrame,columns=columnNames)],ignore_index=True)
			data_Y.append(zeros(len(inputValues)))
			if dataSetParamsDict['scenario'] != 0:  
				dataSetIdCounter += 1
				faulty = normalize(outputData[xNamesFaulty].values,axis=0)
				preDataFrame = vstack([tile(dataSetIdCounter,faulty.shape[0]),array(outputData['time']),faulty.T]).T
				data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrame,columns=columnNames)],ignore_index=True)
				data_Y.append(inputValues)
				embed()
			dataSetIdCounter += 1
			#datasetCutLength = faulty.shape[1]
			'''
		
			#filter implementation (unused)
			#for i in range(len(faulty)):
			#   faulty[i] = savgol_filter(faulty[i],41,3)
			#   nominal[i] = savgol_filter(nominal[i],41,3)
			
			residuals = faulty - nominal
			datasetCutLength = residuals.shape[1]
			preDataFrameResiduals = vstack([tile(dataSetIdCounter,datasetCutLength),arange(datasetCutLength),residuals]).T
			#nominalID = (dataSetID*2)
			#faultyID = dataSetID
			#nonimalStack = vstack([tile(nominalID,datasetCutLength),arange(datasetCutLength),nominal]).T
			#faultyStack = vstack([tile(faultyID,datasetCutLength),arange(datasetCutLength),faulty]).T
			
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
			#outputValues = [np.sqrt(mean_squared_error(nominal.T[i],faulty.T[i])) for i in range(len(xNames))]     #.flatten()
			#resultsList['nominal'] = [nominal, array([0])] #hard coded nominal output
			#if inputValues[0] != 0:
			#data_X = pandas.concat([data_X,pandas.DataFrame(nonimalStack,columns=columnNames)],ignore_index=True)
			#data_Y.append(0)
			#data_X = pandas.concat([data_X,pandas.DataFrame(faultyStack,columns=columnNames)],ignore_index=True)
			data_Y.append(int(dataSetParamsDict['scenario']))
			data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrameResiduals,columns=columnNames)],ignore_index=True)
			dataSetIdCounter += 1
			#data_Y.append(inputValues)
			#data handling
			print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')

		data_Y = pandas.Series(data_Y)
		saveTime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
		data_X.to_csv('X_df_{}.csv'.format(saveTime))
		data_Y.to_csv('y_ds_{}.csv'.format(saveTime))
	except Exception as err:
		print ("Data handling error:", err)
		embed()
if args.fx and args.fy:
	print ('Importing feature datasets - x: {}, y: {}'.format(args.fx, args.fy))
	data_Y = pandas.read_csv(args.fy, index_col=0,squeeze=True) #pandas series
	data_X = pandas.read_csv(args.fx, index_col=0)
else:
	try:
		print ('\nStarting Feature Extraction')
		extractStartTime = time.time()
		FCParameter = 'efficient'
		extraction_settings = {
			'comprehensive': ComprehensiveFCParameters(),
			'efficient': EfficientFCParameters(),
			'minimal': MinimalFCParameters(),
		}
		#X_features = extract_relevant_features(data_X, data_Y, column_id='id', column_sort='time', default_fc_parameters=extraction_settings[FCParameter])
		X_features = extract_features(data_X, column_id='id', column_sort='time', default_fc_parameters=extraction_settings[FCParameter],impute_function=impute)
		saveTime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
		data_X = X_features
		
		data_X.to_csv('X_features_{}.csv'.format(saveTime))
		data_Y.to_csv('y_features_{}.csv'.format(saveTime))
		#saving extracted features
		#https://github.com/zygmuntz/time-series-classification
		#print (X_filtered.info())
		#X_filtered.to_csv('X_{}_{}.csv'.format(FCParameter,saveTime))
		#std = StandardScaler()
		#std.fit(X_filtered_train.values)
		#X_filtered_train = std.transform(X_filtered_train.values)
		#X_filtered_test = std.transform(X_filtered_test.values)
		print ('Feature Extraction Complete!!, it took {} seconds'.format(time.time()-extractStartTime))
	except Exception as err:
		print ("Feature Exraction Error:", err)
		embed()
try:
	'''
	print ('Starting ML Classifier')
	def objective(params,X=X_filtered, Y=y):
		#params = {
		#   'max_depth': int(params['max_depth']),
		#   'gamma': "{:.3f}".format(params['gamma']),
		#   'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
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
	
	X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X_filtered, y, test_size=.4)

	embed()

	estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
							  preprocessing=any_preprocessing('my_pre'),
							  algo=tpe.suggest,
							  max_evals=100,
							  trial_timeout=120)

	# Search the hyperparameter space based on the data

	estim.fit(X_filtered_train, y_train)
	print(estim.score(X_test, y_test))

	print( estim.best_model() )

	estim = HyperoptEstimator( classifier=any_sparse_classifier('clf'), 
							preprocessing=[min_max_scaler('min_max_scaler')],
							algo=tpe.suggest, trial_timeout=300)
	pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),('classifier', estim)])

	pipeline.set_params(augmenter__timeseries_container=X_filtered_train)
	pipeline.fit(X_filtered_train,y_train)
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
	'''
	#from sklearn.neural_network import MLPClassifier
	#clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-3,learning_rate='adaptive', max_iter=1000)
	#clf = MLPClassifier()
	estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
							  preprocessing=any_preprocessing('my_pre'),
							  algo=tpe.suggest,
							  max_evals=150,
							  trial_timeout=120)
	#pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),('classifier', estim)])

	#ppl = Pipeline([
	#	('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
	#	('classifier', ada)
	#  ])

	X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=.20)
	'''
	y_train = y_train.values.reshape(-1,1)
	y_test = y_test.values.reshape(-1,1)
	enc = OneHotEncoder()
	enc.fit(y_train)
	y_train = enc.transform(y_train).toarray()
	y_test = enc.transform(y_test).toarray()
	'''
	print('calculating relevant_features')
	relevant_features = set()
	for label in data_Y.unique():
		y_train_binary = y_train == label
		X_train_filtered = select_features(X_train, y_train_binary)
		print("Number of relevant features for class {}: {}/{}".format(label, X_train_filtered.shape[1], X_train.shape[1]))
		relevant_features = relevant_features.union(set(X_train_filtered.columns))


	# In[ ]:


	print('Length of relevant_features:',len(relevant_features))
	X_train_filtered = X_train[list(relevant_features)]
	X_test_filtered = X_test[list(relevant_features)]

	estim.fit( X_train_filtered, y_train )
	saveTime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	with open("HyperoptEstimator_{}.pkl".format(saveTime), "wb") as f:
		pickle.dump(estim, f)
	X_train.to_csv('X_train_{}.csv'.format(saveTime))
	X_test.to_csv('X_test_{}.csv'.format(saveTime))
	y_train.to_csv('y_train_{}.csv'.format(saveTime))
	y_test.to_csv('y_test_{}.csv'.format(saveTime))
	#with open(eName, "rb") as f:
	#	estim = pickle.load(f)

	# Show the results

	print(estim.score(X_test_filtered, y_test))
	print(estim.best_model())
	clf = estim.best_model()['learner']
	clf.fit(X_train_filtered, y_train)
	y_pred = clf.predict(X_test_filtered)
	print(classification_report(y_test, y_pred))
	print(accuracy_score(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))
	embed()
	#clf = GradientBoostingClassifier()
	ada = AdaBoostClassifier(base_estimator=clf)


	# we keep only those features that we selected above, for both the train and test set

	# In[ ]:
	'''

	X_train_filtered = X_train[list(relevant_features)]
	X_test_filtered = X_test[list(relevant_features)]
	
	X = pandas.DataFrame(index=data_Y.index)
	
	df_ts_train = data_X[data_X["id"].isin(y_train.index)]
	df_ts_test = data_X[data_X["id"].isin(y_test.index)]
	ppl.set_params(augmenter__timeseries_container=df_ts_train);
	ppl.fit(X_train, y_train)
	with open("pipeline.pkl", "wb") as f:
		pickle.dump(ppl, f)
	with open("pipeline.pkl", "rb") as f:
		ppk = pickle.load(f)
	ppk.set_params(augmenter__timeseries_container=df_ts_test);
	y_pred = ppl.predict(X_test)
	print(classification_report(y_test, y_pred))
	print(accuracy_score(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))
	'''

except Exception as err:
	print ("ML Classifier Error:\n")
	exc_type, exc_obj, exc_tb = sys.exc_info()
	fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
	print (exc_type, fname, exc_tb.tb_lineno)
	embed()

embed()