import numpy as np
import os
import matplotlib as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
#import tensorflow as tf
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

outputFolder = "output_625"

listCutFactor = 5
outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = outputDataset[::listCutFactor]

inputDict = {}
outputDict = {}

print ("started data handling")
startTime = time.time()
for data in outputDataset:
	dataPointNum = int(data.split('_')[0])
	inputDict[dataPointNum] = np.array([float(i) for i in data.replace('.csv','').split('_')[1:]])
	outputDict[dataPointNum] = pd.read_csv(outputFolder+"/"+data)
print "finished data handling: {:.4}s".format(time.time()-startTime)

xNames = ['q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','q4_faulty','omega1_healthy','omega2_healthy','omega3_healthy']
#xNames = ['time','q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty','wheel1_i_faulty','wheel1_w_faulty','wheel2_i_faulty','wheel2_w_faulty','wheel3_i_faulty','wheel3_w_faulty',\
#	'wheel4_i_faulty','wheel4_w_faulty']

#xValues = None
#yValues = None
xValues = []
yValues = []

startTime = time.time()
print ("starting data conversion to 2d array for split into train yvalues")
sets = list(outputDict.keys())#[:len(outputDict.keys()):10]
for key in sets:#[:len(outputDict.keys())//10:5]: #[:N]
	outputValues = outputDict[key][xNames].values#.flatten()
	inputValues = inputDict[key][0:1]
	xValues.append(outputValues)
	yValues.append(inputValues)
	#yValues = inputValues if yValues is None else np.vstack((yValues,inputValues))
	#xValues = outputValues if xValues is None else np.vstack((xValues,outputValues))
print "ended data conversion to 2d array for split into train yvalues: {:.4}s".format(time.time()-startTime)
yValues = np.ravel(yValues)
xValues = np.array(xValues)
X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=123) #explain hypervariables
#X_train = normalize(X_train)
#X_test = normalize(X_test)
from IPython import embed; embed()
from tslearn.svm import TimeSeriesSVC
#clf = TimeSeriesSVC(kernel="poly", gamma=0.1, sz=X_train.shape[1], d=X_train.shape[2], verbose=True)
#clf = AdaBoostClassifier(TimeSeriesSVC(kernel="poly", sz=X_train.shape[1], d=X_train.shape[2],verbose=True))
clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=1000,n_jobs=-1, verbose=True))
startTime = time.time()
clf.fit(X_train,y_train)
print "fit time: {}".format(time.time()-startTime)
y_pred = clf.predict(X_test)
print accuracy_score(y_pred, y_test)
print metrics.confusion_matrix(y_test, y_pred, labels=range(16))

'''
clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw").fit(X_train, y_train)
clf = AdaBoostClassifier(TimeSeriesSVC(kernel="poly", gamma=.1, sz=X_train.shape[1], d=X_train.shape[2],verbose=True))
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)
metrics.confusion_matrix(y_test, y_pred, labels=range(16))
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
'''