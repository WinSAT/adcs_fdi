'''
train with nominal and fault seperately (non-residual) 
use current from each rw
'''

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
from scipy.signal import savgol_filter


outputFolder = "output_300_constSeverity_csvs"

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]

listCutFactor = 1
if listCutFactor > 1:
	outputDataset = outputDataset[::listCutFactor]

timeframe = 10
csvTimestep = 0.1

inputDict = {}
outputDict = {}

xNamesFaulty = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty','wheel1_i_faulty','wheel2_i_faulty','wheel3_i_faulty','wheel4_i_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy','wheel1_i_healthy','wheel2_i_healthy','wheel3_i_healthy','wheel4_i_healthy']
xNamesTrain = ['id', 'time'] + ['_'.join(i.split('_')[:-1]) for i in xNamesFaulty]
#xNames = ['q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty','wheel1_i_faulty','wheel1_w_faulty','wheel2_i_faulty','wheel2_w_faulty','wheel3_i_faulty','wheel3_w_faulty',\
#	'wheel4_i_faulty','wheel4_w_faulty']
#xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy','wheel1_i_healthy','wheel1_w_healthy','wheel2_i_healthy','wheel2_w_healthy','wheel3_i_healthy','wheel3_w_healthy',\
#	'wheel4_i_healthy','wheel4_w_healthy']

#num,scenario,kt,vbus,ktInception,vbusInception,ktDuration,vbusDuration,ktSeverity,vbusSeverity
#xValues = None
#yValues = None
xValues = []
yValues = []


print ("started data handling")
startTime = time.time()


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

allDataX = {}
allDataY = []
idCounter = -1
for dataSetId,data in enumerate(outputDataset):
	dataSetParams = data.replace('.csv','').split('_')
	dataPointNum = int(dataSetParams[0])
	scenarioNum = int(dataSetParams[1])
	#if scenarioNum > 4:
	#	continue
	inputData = array([float(i) for i in dataSetParams[1:]])

	#if inputData[1] != inputData[2]:
	#	continue

	#inceptionTime = int(inputData[3]/csvTimestep)
	#timeframe = int(inputData[5]) if int(inputData[5]) >= 2 else 1
	#endTime = int((inputData[3]+timeframe)/csvTimestep)
	#targetTime = 30
	#deviation = 5	
	#if int(inputData[3]) > targetTime+deviation or int(inputData[3]) < targetTime-deviation and scenarioNum != 0:
	#	continue

	#if inputData[1] and inputData[1]

	outputData = pd.read_csv(outputFolder+"/"+data)

	#set rsme values  transformer.fit_transform(as features
	#embed()
	#num,scenario,kt,vbus,ktInception,vbusInception,ktDuration,vbusDuration,ktSeverity,vbusSeverity
	inputValues = take(inputData,[0]).astype(int)
	#inputValues = 1 if inputData[0] != 0 else 0
	#inputValues = np.array([inputValues])

	#int(inputData[5])
	#faulty = normalize(outputData[xNames].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	#nominal = normalize(outputData[xNamesNominal].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	faulty = (outputData[xNamesFaulty].values).T
	nominal =  (outputData[xNamesNominal].values).T

	for i in range(len(faulty)):
		faulty[i] = savgol_filter(faulty[i],41,3)
		nominal[i] = savgol_filter(nominal[i],41,3)

	residuals = faulty - nominal
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
		embed()

	#nominal = StandardScaler().fit_transform(nominal)
	#faulty = StandardScaler().fit_transform(faulty)
	#outputValues = [np.sqrt(mean_squared_error(nominal.T[i],faulty.T[i])) for i in range(len(xNames))] 	#.flatten()
	resultsList = {}
	#resultsList['nominal'] = [nominal, array([0])] #hard coded nominal output
	#if inputValues[0] != 0:
	resultsList['residuals'] = [residuals, inputValues]
	trainSetX = []
	trainSetY = []
	for dataType, currentDataset in resultsList.items():
		idCounter += 1
		currentTrainSetX = currentDataset[0]
		currentTrainSetY = currentDataset[1]
		scenarioData = (vstack((tile([idCounter],len(currentTrainSetX.T)),arange(len(currentTrainSetX.T)),currentTrainSetX))).tolist()
		for col in xNamesTrain:
			allDataX.setdefault(col,[]).append(scenarioData[xNamesTrain.index(col)])
		allDataY.append(currentTrainSetY[0])
		#if addIdxNum == 1:
		#	embed()
		#xValues.append(array([generateTimeseriesFeatures(ts) for ts in currentTrainSetX]).ravel())
		#yValues.append(currentTrainSetY)
	#yValues = inputValues if yValues is None else np.vstack((yValues,inputValues))
	#xValues = outputValues if xValues is None else np.vstack((xValues,outputValues))
allDataX = pd.DataFrame({k:array(val).flatten() for k,val in allDataX.items()})
allDataY = pd.Series(allDataY)
print 'done DataHandling'
from tsfresh import extract_relevant_features
#extraction_settings = ComprehensiveFCParameters()
#extraction_settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
extraction_settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
features_filtered_direct = extract_relevant_features(allDataX, allDataY,column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
try:
	X_train, X_test, y_train, y_test = train_test_split(features_filtered_direct, allDataY, test_size=.2)
	cl = AdaBoostClassifier(RandomForestClassifier())
	cl.fit(X_train, y_train)
	print classification_report(y_test, cl.predict(X_test)) 
	print cl.n_features_ 
except:
	embed()
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