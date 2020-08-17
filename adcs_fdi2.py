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
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import time
from IPython import embed

import pandas as pd
import argparse

from tsfresh import extract_features, extract_relevant_features, select_features
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


outputFolder = "output_625"
stepsizeFreq = 10.0

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
columnNames = ['id','time']+[i.split('_')[0] for i in xNamesFaulty]

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

print ('Starting Data Handling')
#data_X = pd.DataFrame([],columns=columnNames)
data_X = []
data_Y = []
chuckSize = 40;
dataSetIDCounter = []
csvFilenameX = 'features_X.csv'
csvFilenameY = 'features_Y.csv'

if not os.path.isfile(csvFilenameX) and not os.path.isfile(csvFilenameY):
	print_progress(0, len(outputDataset), prefix = 'Data Handling:')
	for dataSetID,data in enumerate(outputDataset):
		dataSetIDCounter.append(dataSetID)
		dataSetParams = data.replace('.csv','').split('_')
		dataSetParamsDict = {}
		#get dataset parameters as dict
		for idx,paramName in enumerate(["id","scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration", "ktSeverity", "vbusSeverity"]):
			dataSetParamsDict[paramName] = float(dataSetParams[idx])
		#dataset parameters as array excluding id num
		inputData = array([float(i) for i in dataSetParams[1:]])
		outputData = pd.read_csv(outputFolder+"/"+data)

		inputValues = array([int(dataSetParamsDict[i]) for i in ["scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration"]])

		faulty = outputData[xNamesFaulty].values.T
		nominal =  outputData[xNamesNominal].values.T
		residual = (nominal-faulty)
		residualDF = vstack([tile(dataSetID,residual.shape[1]),arange(residual.shape[1]),residual]).T
		residualDF = pd.DataFrame(residualDF,columns=columnNames)
		resFeatures = extract_features(residualDF, column_id="id", column_sort="time",default_fc_parameters=feature_settings,disable_progressbar=True)
		if len(data_X) == 0:
			data_X = empty((0,len(resFeatures.values[0])), float)
		data_X = vstack((data_X,resFeatures.values[0]))
		data_Y.append(dataSetParamsDict['scenario'])
		print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')

		if dataSetID%chuckSize == 0:
			df_data_X = pd.DataFrame(data=data_X,index=dataSetIDCounter,columns=resFeatures.columns)
			df_data_Y = pd.DataFrame(data=data_Y,index=dataSetIDCounter,columns=['scenario'])
			if os.path.isfile(csvFilenameX) and os.path.isfile(csvFilenameY):
				df_data_X.to_csv(csvFilenameX, mode='a', header=False)
				df_data_Y.to_csv(csvFilenameY, mode='a', header=False)
			else:
				df_data_X.to_csv(csvFilenameX, mode='w')
				df_data_Y.to_csv(csvFilenameY, mode='w')
			data_X = []
			data_Y = []
			dataSetIDCounter = []
else:
	print('Importing Data from: {}, {}'.format(csvFilenameX,csvFilenameY))

data_X = pd.read_csv(csvFilenameX).drop('Unnamed: 0',axis=1)
data_Y = pd.read_csv(csvFilenameY).drop('Unnamed: 0',axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

X_train, X_test, y_train, y_test = train_test_split(data_X.values, data_Y.values.T[0].reshape(-1, 1), test_size=0.20)

enc = OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

y_train_1d = np.ravel(enc.inverse_transform(y_train))
y_test_1d = np.ravel(enc.inverse_transform(y_test))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

args = {
    'n_estimators':200,
    'max_depth':25
}

clf = RandomForestClassifier(**args)

n_feats = np.linspace(0.05, 1, 20)
print(n_feats)
train_size_abs, train_scores, test_scores = learning_curve(clf, X_train, y_train, train_sizes=n_feats, n_jobs=-1, verbose=10, cv=5)


def plot_learning_curve(train_size_abs, train_scores, test_scores):
    train_score = train_scores.mean(axis = 1)
    train_std = train_scores.std(axis = 1)
    test_score = test_scores.mean(axis = 1)
    test_std = test_scores.std(axis = 1)

    plt.rcParams["figure.figsize"] = (12,10)
    plt.rcParams["font.size"] = 25
    sn.set_style("whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.style.use('grayscale')

    plt.scatter(train_size_abs, train_score, s=100, marker="o", label="Train Set")
    plt.plot(train_size_abs, train_score, "--")
    # plt.fill_between(train_size_abs, train_score + train_std, train_score - train_std)

    plt.scatter(train_size_abs, test_score, s=100, marker="^", label="Test Set")
    plt.plot(train_size_abs, test_score, "--")
    # plt.fill_between(train_size_abs, test_score + 2 * test_std, test_score - 2 * test_std, alpha=0.2)
    plt.legend()
    plt.xlabel("Training Set Size")
    plt.tight_layout()

plot_learning_curve(train_size_abs, train_scores, test_scores)
plt.ylabel("Classifier Accuracy Score (k=5)")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier

args = {
    'n_estimators':200,
    'max_depth':50
}

clf = RandomForestClassifier(**args)
clf.fit(X_train, y_train_1d)

rfe_selector = RFE(clf, n_features_to_select=300, step=0.05, verbose=1)
rfe_selector = rfe_selector.fit(X_train, y_train_1d)

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

args = {
    'n_estimators':200,
}

clf_randomforest = RandomForestClassifier(**args)

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf_randomforest, step=15, cv=10, min_features_to_select=5,
              scoring='accuracy', verbose=10)
rfecv.fit(X_train, y_train_1d)

for y, x in zip(rfecv2.grid_scores_, x_feat):
    print("{} {}".format(x, y))

# Plot number of features VS. cross-validation scores
plt.xlabel("Number of features selected")
plt.ylabel("Classifier Accuracy Score (k=10)")
plt.scatter(x_feat, rfecv2.grid_scores_, s=70)

"""
Different trendlines to visualize the results
"""

from scipy.optimize import curve_fit

def log(x, a, b, c):
    return a*np.log(b*x) + c

def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = curve_fit(log, x_feat, rfecv2.grid_scores_)

xx = np.arange(10, 660, 1)
yy = log(xx, *popt)
plt.plot(xx, yy, "b--")
xx = np.arange(0.01, 10, 0.01)
yy = log(xx, *popt)
plt.plot(xx, yy, "b--")

plt.show()




embed()