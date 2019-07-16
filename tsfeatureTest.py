import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython import embed

from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from tsfresh.feature_extraction import feature_calculators
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

outputFolder = "output_300_constSeverity"

listCutFactor = 2
outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = outputDataset[::listCutFactor]

timeframe = 10
csvTimestep = 0.1

inputDict = {}
outputDict = {}

xNames = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']

xValues = []
yValues = []

tsFeatureDict = {
	'abs_energy': [],# 	Returns the absolute energy of the time series which is the sum over the squared values
	'absolute_sum_of_changes': [],# 	Returns the sum over the absolute value of consecutive changes in the series x
	#'agg_linear_trend': [param],# 	Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus the sequence from 0 up to the number of chunks minus one.
	'approximate_entropy': [5, 3],# 	Implements a vectorized Approximate entropy algorithm.
	#'ar_coefficient': [param],# 	This feature calculator fits the unconditional maximum likelihood of an autoregressive AR(k],# process.
	#'augmented_dickey_fuller': [param],# 	The Augmented Dickey-Fuller test is a hypothesis test which checks whether a unit root is present in a time series sample.
	#'autocorrelation': [lag],# 	Calculates the autocorrelation of the specified lag, according to the formula [1],
	#'binned_entropy': [max_bins],# 	First bins the values of x into max_bins equidistant bins.
	#'c3': [lag],# 	This function calculates the value of
	#'change_quantiles': [ql, qh, isabs, f_agg],# 	First fixes a corridor given by the quantiles ql and qh of the distribution of x.
	#'cid_ce': [normalize],# 	This function calculator is an estimate for a time series complexity [1], (A more complex time series has more peaks, valleys etc.],#.
	'count_above_mean': [],# 	Returns the number of values in x that are higher than the mean of x
	'count_below_mean': [],# 	Returns the number of values in x that are lower than the mean of x
	#'cwt_coefficients': [param],
	#'energy_ratio_by_chunks': [param],# 	Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole series.
	#'fft_aggregated': [param],# 	Returns the spectral centroid (mean],#, variance, skew, and kurtosis of the absolute fourier transform spectrum.
	#'fft_coefficient': [param],# 	Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
	'first_location_of_maximum': [],# 	Returns the first location of the maximum value of x.
	'first_location_of_minimum': [],# 	Returns the first location of the minimal value of x.
	#'friedrich_coefficients': [param],# 	Coefficients of polynomial h(x],#, which has been fitted to
	'has_duplicate': [],# 	Checks if any value in x occurs more than once
	'has_duplicate_max': [],# 	Checks if the maximum value of x is observed more than once
	'has_duplicate_min': [],# 	Checks if the minimal value of x is observed more than once
	#'index_mass_quantile': [param],# 	Those apply features calculate the relative index i where q% of the mass of the time series x lie left of i.
	'kurtosis': [],# 	Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2],#.
	#'large_standard_deviation': [r],
	'last_location_of_maximum': [],# 	Returns the relative last location of the maximum value of x.
	'last_location_of_minimum': [],# 	Returns the last location of the minimal value of x.
	'length': [],# 	Returns the length of x
	#'linear_trend': [param],# 	Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one.
	#'linear_trend_timewise': [param],# 	Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to length of the time series minus one.
	'longest_strike_above_mean': [],# 	Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x
	'longest_strike_below_mean': [],# 	Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x
	#'max_langevin_fixed_point': [r, m],# 	Largest fixed point of dynamics :math:argmax_x {h(x],#=0}` estimated from polynomial h(x],#,
	'maximum': [],# 	Calculates the highest value of the time series x.
	'mean': [],# 	Returns the mean of x
	'mean_abs_change': [],# 	Returns the mean over the absolute differences between subsequent time series values which is
	'mean_change': [],# 	Returns the mean over the differences between subsequent time series values which is
	'mean_second_derivative_central': [],# 	Returns the mean value of a central approximation of the second derivative
	'median': [],# 	Returns the median of x
	'minimum': [],# 	Calculates the lowest value of the time series x.
	#'number_crossing_m': [m],# 	Calculates the number of crossings of x on m.
	#'number_cwt_peaks': [n],# 	This feature calculator searches for different peaks in x.
	#'number_peaks': [n],# 	Calculates the number of peaks of at least support n in the time series x.
	#'partial_autocorrelation': [param],# 	Calculates the value of the partial autocorrelation function at the given lag.
	'percentage_of_reoccurring_datapoints_to_all_datapoints': [],# 	Returns the percentage of unique values, that are present in the time series more than once.
	'percentage_of_reoccurring_values_to_all_values': [],# 	Returns the ratio of unique values, that are present in the time series more than once.
	#'quantile': [q],# 	Calculates the q quantile of x.
	#'range_count': [min, max],# 	Count observed values within the interval [min, max],#.
	#'ratio_beyond_r_sigma': [r],# 	Ratio of values that are more than r*std(x],# (so r sigma],# away from the mean of x.
	'ratio_value_number_to_time_series_length': [],# 	Returns a factor which is 1 if all values in the time series occur only once, and below one if this is not the case.
	'sample_entropy': [],# 	Calculate and return sample entropy of x.
	#'set_property': [key, value],# 	This method returns a decorator that sets the property key of the function to value
	'skewness': [],# 	Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G1],#.
	#'spkt_welch_density': [param],# 	This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
	'standard_deviation': [],# 	Returns the standard deviation of x
	'sum_of_reoccurring_data_points': [],# 	Returns the sum of all data points, that are present in the time series more than once.
	'sum_of_reoccurring_values': [],# 	Returns the sum of all values, that are present in the time series more than once.
	'sum_values': [],# 	Calculates the sum over the time series values
	#'symmetry_looking': [param],# 	Boolean variable denoting if the distribution of x looks symmetric.
	#'time_reversal_asymmetry_statistic': [lag],# 	This function calculates the value of
	#'value_count': [value],# 	Count occurrences of value in time series x.
	'variance': [],# 	Returns the variance of x
	'variance_larger_than_standard_deviation': [],#
}

print ("started data handling")
startTime = time.time()
def sav(y,n=51,d=3):
	return savgol_filter(y,n,d)

X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = []
y_test = []
count = 0
for data in outputDataset:
	dataPointNum = int(data.split('_')[0])
	scenarioNum = int(data.split('_')[1])
	if scenarioNum > 4:
		continue
	inputData = np.array([float(i) for i in data.replace('.csv','').split('_')[1:]])

	#if inputData[1] != inputData[2]:
	#	continue

	inceptionTime = int(inputData[3]/csvTimestep)
	timeframe = int(inputData[5]) if int(inputData[5]) >= 2 else 1
	endTime = int((inputData[3]+timeframe)/csvTimestep)
	#targetTime = 30
	#deviation = 5
	#if int(inputData[3]) > targetTime+deviation or int(inputData[3]) < targetTime-deviation and scenarioNum != 0:
	#	continue

	#if inputData[1] and inputData[1]

	outputData = pd.read_csv(outputFolder+"/"+data)

	#set rsme values  transformer.fit_transform(as features
	inputValues = np.take(inputData,[0]).astype(int)
	#inputValues = 1 if inputData[0] != 0 else 0
	#inputValues = np.array([inputValues])

	#int(inputData[5])
	#faulty = normalize(outputData[xNames].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	#nominal = normalize(outputData[xNamesNominal].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	faulty = (outputData[xNames].values)[inceptionTime:endTime]
	nominal =  (outputData[xNamesNominal].values)[inceptionTime:endTime]
	#residual = faulty - nominal
	residual = faulty - nominal

	if count % 10 < 2:
		scenarioFeatureMap = {'id':np.tile(len(y_test),len(residual)),'time':np.arange(len(residual))*csvTimestep}
		scenarioFeatureMap.update({col.replace('faulty','residual'):residual.T[idx] for idx,col in enumerate(xNames)})
		X_test = X_test.append(pd.DataFrame(scenarioFeatureMap),ignore_index=1)
		y_test.append(scenarioNum)
	else:
		scenarioFeatureMap = {'id':np.tile(len(y_train),len(residual)),'time':np.arange(len(residual))*csvTimestep}
		scenarioFeatureMap.update({col.replace('faulty','residual'):residual.T[idx] for idx,col in enumerate(xNames)})
		X_train = X_train.append(pd.DataFrame(scenarioFeatureMap),ignore_index=1)
		y_train.append(scenarioNum)

	count += 1
	#embed()
	#xValues.append(outputValuesQ)
	#yValues.append(inputValues)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
print "finished data handling: {:.4}s".format(time.time()-startTime)
#X = extract_features(trainFeatureMap,column_id='id', column_sort='time', default_fc_parameters=EfficientFCParameters())

pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
            ('classifier', RandomForestClassifier())])
Xtrain = pd.DataFrame(index=y_train.index)
pipeline.set_params(augmenter__timeseries_container=X_train)
pipeline.fit(Xtrain, y_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
scenarioNumMax = 5
pipeline.set_params(augmenter__timeseries_container=X_test)
Xtest = pd.DataFrame(index=y_test.index)
y_pred = pipeline.predict(Xtest)
print classification_report(y_test.as_matrix(), y_pred)
cm = confusion_matrix(y_test.as_matrix(), y_pred, labels=range(scenarioNumMax))
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print cm
print accuracy_score(y_pred,y_test)
df_cm = pd.DataFrame(cm, range(scenarioNumMax),range(scenarioNumMax))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()

embed()


