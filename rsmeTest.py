import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython import embed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
#import tensorflow as tf
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

from scipy.stats import skew
from scipy.stats import kurtosis
from statsmodels import robust

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from tsfresh.feature_extraction import feature_calculators

outputFolder = "output_300_constSeverity"

listCutFactor = 5
outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = outputDataset[::listCutFactor]

timeframe = 10
csvTimestep = 0.1

inputDict = {}
outputDict = {}

xNames = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
#xNames = ['q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty','wheel1_i_faulty','wheel1_w_faulty','wheel2_i_faulty','wheel2_w_faulty','wheel3_i_faulty','wheel3_w_faulty',\
#	'wheel4_i_faulty','wheel4_w_faulty']
#xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy','wheel1_i_healthy','wheel1_w_healthy','wheel2_i_healthy','wheel2_w_healthy','wheel3_i_healthy','wheel3_w_healthy',\
#	'wheel4_i_healthy','wheel4_w_healthy']

#xValues = None
#yValues = None
xValues = []
yValues = []

tsFeatureDict = {
	'abs_energy': [],# 	Returns the absolute energy of the time series which is the sum over the squared values
	'absolute_sum_of_changes': [],# 	Returns the sum over the absolute value of consecutive changes in the series x
	#'agg_linear_trend': [param],# 	Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus the sequence from 0 up to the number of chunks minus one.
	#'approximate_entropy': [5, 3],# 	Implements a vectorized Approximate entropy algorithm.
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
	#nominal = StandardScaler().fit_transform(nominal)
	#faulty = StandardScaler().fit_transform(faulty)
	#outputValues = [np.sqrt(mean_squared_error(nominal.T[i],faulty.T[i])) for i in range(len(xNames))] 	#.flatten()
	outputValuesQ = []
	outputValuesW = []
	for i in range(len(xNames)):
		try:
			n = len(nominal.T[i])-1 if len(nominal.T[i]) % 2 == 0 else len(nominal.T[i])
			#nom = sav(nominal.T[i],n=n)
			#fal = sav(faulty.T[i],n=n)
			nom = nominal.T[i]
			fal = faulty.T[i]
			#nom,fal = normalize([nom,fal])
			residual = nom-fal
			#embed()
			for method,params in tsFeatureDict.items():
				try:
					outputValuesQ.append(getattr(feature_calculators, method)(residual))
				except:
					print "error on ", method
					embed()
			#outputValuesQ.append(np.sqrt(mean_squared_error(nom,fal)))
			#outputValuesQ.append(np.max(residual))
			#outputValuesQ.append(np.std(residual))
			#outputValuesQ.append(np.min(residual))
			#outputValuesQ.append(np.mean(residual))
			#outputValuesQ.append(skew(residual))
			#outputValuesQ.append(kurtosis(residual))
			#outputValuesQ.append(robust.mad(residual))
			#outputValuesQ.append(np.max(residual)/np.sqrt(mean_squared_error(nom,fal)))
			#outputValuesQ.append(np.sqrt(mean_squared_error(nom,fal))/np.mean(residual))
			#outputValuesQ.append(np.max(residual)/np.mean(residual))
			#outputValuesQ.append(mean_absolute_error(nom,fal))
		except Exception as e: 
			print "Error!:", e,"\n"
			embed()
		#outputValues.append(mean_absolute_error(nominal.T[i]**p,faulty.T[i]**p))
		#outputValues.append(r2_score(nominal.T[i]**p,faulty.T[i]**p))
		#outputValues.append(explained_variance_score(nominal.T[i]**p,faulty.T[i]**p))
		#outputValues.append(mean_absolute_error(nominal.T[i]**p,faulty.T[i]**p))

	#from IPython import embed; embed()
	xValues.append(outputValuesQ)
	yValues.append(inputValues)
	#yValues = inputValues if yValues is None else np.vstack((yValues,inputValues))
	#xValues = outputValues if xValues is None else np.vstack((xValues,outputValues))
print "finished data handling: {:.4}s".format(time.time()-startTime)

yValues = np.ravel(yValues)
#yValues = np.array(yValues)
xValues = np.array(xValues)
from IPython import embed; embed()
X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=123) #explain hypervariables


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
'''
import colorsys
N = 16
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
'''

cc = [(0.5, 0.25, 0.25),
 (0.5, 0.34375, 0.25),
 (0.5, 0.4375, 0.25),
 (0.46875, 0.5, 0.25),
 (0.375, 0.5, 0.25),
 (0.28125, 0.5, 0.25),
 (0.25, 0.5, 0.3125),
 (0.25, 0.5, 0.40625),
 (0.25, 0.5, 0.5),
 (0.25, 0.40625, 0.5),
 (0.25, 0.3125, 0.5),
 (0.28125, 0.25, 0.5),
 (0.375, 0.25, 0.5),
 (0.46875, 0.25, 0.5),
 (0.5, 0.25, 0.4375),
 (0.5, 0.25, 0.34375)]

cc = [(0.5, 0.25, 0.25),
 (0.44999999999999996, 0.5, 0.25),
 (0.25, 0.5, 0.3500000000000001),
 (0.25, 0.3500000000000001, 0.5),
 (0.4500000000000002, 0.25, 0.5)]


#plt.show()

#pcatest = pca.fit_transform(X_test)
#v = #KMeans(n_clusters=5,random_state=123)
n_components = 4
#v = GaussianMixture(n_components=n_components, covariance_type='full')
#y_pred_v = v.fit_predict(X_train)
#y_pred_v = v.fit_predict(pcatst)
#clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
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
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print cm
		rfpred = y_pred

csfont = {'fontname':'Times New Roman'}


pca = PCA(n_components=3)
pcatrain = pca.fit_transform(X_train)
ax = plt.figure().add_subplot(111, projection='3d')
from IPython import embed; embed()
plt.rcParams.update({'font.serif': 'Times New Roman'})
for idx,i in enumerate(X_test):
	ax.scatter(*pcatrain[idx], color=cc[rfpred[idx]])
#ax.text2D(0.05, 0.95, "PCA 3D Values - Colored by Predicted Scenario Num",transform=ax.transAxes)
ax.set_xlabel('$PCA_1$ axis', **csfont)
ax.set_ylabel('$PCA_2$ axis', **csfont)
ax.set_zlabel('$PCA_3$ axis', **csfont)
plt.savefig('PCA_3D_Xtrain_Predict.svg', format='svg', dpi=1000)
plt.clf()
ax2 = plt.figure().add_subplot(111, projection='3d')
for idx,i in enumerate(X_test):
	ax2.scatter(*pcatrain[idx], color=cc[y_test[idx]])
#ax2.text2D(0.05, 0.95, "PCA 3D Values - Colored by Actual Scenario Num", transform=ax2.transAxes)
ax2.set_xlabel('$PCA_1$ axis', **csfont)
ax2.set_ylabel('$PCA_2$ axis', **csfont)
ax2.set_zlabel('$PCA_3$ axis', **csfont)
#plt.title('title',**csfont)
plt.savefig('PCA_3D_Xtrain_Actual.svg', format='svg', dpi=1000)

from IPython import embed; embed()
'''
ax = plt.figure().add_subplot(111, projection='3d')
ax2 = plt.figure().add_subplot(111, projection='3d')
[ax.scatter(*pcatest[i],color=cc[y_test[i][0]]) for i in range(len(pcatest))]
[ax2.scatter(*pcatrain[i],color=cc[y_train[i][0]]) for i in range(len(pcatrain))]
plt.show()
for i in range(len(xValues)):
	plt.scatter(*xValues[i],color=cc[yValues[i]])
plt.show()
#for i in range(len(pcatst)):
#	plt.scatter(*pcatst[i], color=cc[y_train[i]])
#plt.show()

plt.close()
fig = plt.figure()
#fig2 = plt.figure()
fig3 = plt.figure()
#fig4 = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax2 = fig2.add_subplot(111, projection='3d')
ax3 = fig3.add_subplot(111, projection='3d')
#ax4 = fig4.add_subplot(111, projection='3d')
fig.suptitle("normalized rsme - q1,q1,q3 - colored by scenario")
#fig2.suptitle("normalized rsme - omega1,omega1,omega3 - colored by scenario")
fig3.suptitle("normalized rsme - q1,q1,q3 - colored by kmeans cluster num")
#fig4.suptitle("normalized rsme - omega1,omega1,omega3 - colored by kmeans cluster num")
from IPython import embed; embed()
#xtrain = X_train
xtrain = pcatst
for i in range(len(xtrain)):
	#from IPython import embed; embed()
	ax.scatter(xtrain[i][0],xtrain[i][1],xtrain[i][2],color=cc[y_train[i]])
	#ax2.scatter(X_train[i][3],X_train[i][4],X_train[i][5],color=cc[y_train[i]])
	ax3.scatter(xtrain[i][0],xtrain[i][1],xtrain[i][2],color=cc[y_pred_v[i]])
	#ax4.scatter(X_train[i][3],X_train[i][4],X_train[i][5],color=cc[y_pred_v[i]])
plt.show()
'''

clusters = {
	"SpectralClustering": SpectralClustering(n_clusters=n_components,assign_labels="discretize",random_state=123),
	"KMeans": KMeans(n_clusters=n_components,random_state=123),
	#"GaussianMixture": GaussianMixture(n_components=n_components, covariance_type='full')
}

clfs = {
	"DecisionTree": DecisionTreeClassifier(max_depth=5),
	"KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
	"AdaBoostClassifier": AdaBoostClassifier(),
	"AdaBoostClassifier RF": AdaBoostClassifier(RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)),
	"MLPClassifier": MLPClassifier(alpha=1, max_iter=1000),
}
print "Accuracy scores:"
for cluName, clu in clusters.items():
	for clfName, clf in clfs.items():
		y_pred_clu = clu.fit_predict(X_train)
		clf.fit(y_pred_clu.reshape(-1,1),y_train)
		y_pred = clf.predict(clu.fit_predict(X_test).reshape(-1,1)) 
		print "{} & {} : {:.3}%".format(cluName,clfName,accuracy_score(y_pred, y_test)*100)
		clf = clone(clf)
	clu = clone(clu)