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

import pandas
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
#data_X = pandas.DataFrame([],columns=columnNames)
data_X = []
data_Y = []
chuckSize = 40;
dataSetIDCounter = []
csvFilenameX = 'features_X.csv'
csvFilenameY = 'features_Y.csv'

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
	outputData = pandas.read_csv(outputFolder+"/"+data)

	inputValues = array([int(dataSetParamsDict[i]) for i in ["scenario","kt", "vbus", "ktInception", "vbusInception","ktDuration", "vbusDuration"]])

	faulty = outputData[xNamesFaulty].values.T
	nominal =  outputData[xNamesNominal].values.T
	residual = (nominal-faulty)
	residualDF = vstack([tile(dataSetID,residual.shape[1]),arange(residual.shape[1]),residual]).T
	residualDF = pandas.DataFrame(residualDF,columns=columnNames)
	resFeatures = extract_features(residualDF, column_id="id", column_sort="time",default_fc_parameters=feature_settings,disable_progressbar=True)
	if len(data_X) == 0:
		data_X = empty((0,len(resFeatures.values[0])), float)
	data_X = vstack((data_X,resFeatures.values[0]))
	data_Y.append(dataSetParamsDict['scenario'])
	print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')

	if dataSetID%chuckSize == 0:
		df_data_X = pandas.DataFrame(data=data_X,index=dataSetIDCounter,columns=resFeatures.columns)
		df_data_Y = pandas.DataFrame(data=data_Y,index=dataSetIDCounter,columns=['scenario'])
		if os.path.isfile(csvFilenameX) and os.path.isfile(csvFilenameY):
			df_data_X.to_csv(csvFilenameX, mode='a', header=False)
			df_data_Y.to_csv(csvFilenameY, mode='a', header=False)
		else:
			df_data_X.to_csv(csvFilenameX, mode='w')
			df_data_Y.to_csv(csvFilenameY, mode='w')
		data_X = []
		data_Y = []
		dataSetIDCounter = []

embed()