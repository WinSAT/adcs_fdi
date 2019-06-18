import numpy as np
import sklearn
import os
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize
import time
from termcolor import colored

outputFolder = "output_625"
trainRatio = 0.8

#csvName = "0_0_0_0_0_0_0_0_0.029_6.csv"

listCutFactor = 5
outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = outputDataset[::listCutFactor]
from IPython import embed; embed()
#outputDataset = sorted(outputDataset, key=lambda i: float(i.split("_")[0]))

inputDict = {}
outputDict = {}

print ("started data handling")
startTime = time.time()
for data in outputDataset:
	dataPointNum = int(data.split('_')[0])
	inputDict[dataPointNum] = np.array([float(i) for i in data.replace('.csv','').split('_')[1:]])
	outputDict[dataPointNum] = pd.read_csv(outputFolder+"/"+data)
print "finished data handling: {:.4}s".format(colored(time.time()-startTime,'red'))

headers = ['time','q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy',\
	'wheel1_i_healthy','wheel1_w_healthy','wheel2_i_heatlhy','wheel2_w_healthy','wheel3_i_healthy','wheel3_w_healthy',\
	'wheel4_i_healthy','wheel4_w_healthy','q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty',\
	'omega3_faulty','wheel1_i_faulty','wheel1_w_faulty','wheel2_i_faulty','wheel2_w_faulty','wheel3_i_faulty','wheel3_w_faulty',\
	'wheel4_i_faulty','wheel4_w_faulty']

xNames = ['time','q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty']

xNameNominal = ['time','q1_healthy','q2_healthy','q3_healthy','q4_healthy','omega1_healthy','omega2_healthy','omega3_healthy',\
	'wheel1_i_healthy','wheel1_w_healthy','wheel2_i_healthy','wheel2_w_healthy','wheel3_i_healthy','wheel3_w_healthy',\
	'wheel4_i_healthy','wheel4_w_healthy']

xValues = None
yValues = None
#from IPython import embed; embed()
startTime = time.time()
print ("starting data conversion to 2d array for split into train yvalues")
sets = list(outputDict.keys())#[:len(outputDict.keys()):10]
for key in sets:#[:len(outputDict.keys())//10:5]: #[:N]
	#from IPython import embed; embed()
	outputValues = outputDict[key][xNames].values.flatten()
	#outVals = outputDict[key][xNames].values
	#outputValues = np.vstack((outVals.T[0],normalize((outVals.T[1:].T)).T)).flatten()
	#outputValues = np.concatenate((outputDict[key][xNames].values.T,((outputDict[key][xNames].values).T[1:]**2))).T.flatten()
	inputValues = inputDict[key][0:1]
	#outputValuesNominal = outputDict[key][xNames].values.flatten()
	#inputValuesNominal = [0]
	#outputValues = outputDict[key][xNames].values
	#inputValues = np.tile(inputDict[key].astype(int),(outputValues.shape[0],1))
	yValues = inputValues if yValues is None else np.vstack((yValues,inputValues))
	xValues = outputValues if xValues is None else np.vstack((xValues,outputValues))
	#yValues = np.vstack((yValues,inputValuesNominal))
	#xValues = np.vstack((xValues,outputValuesNominal))
print "ended data conversion to 2d array for split into train yvalues: {:.4}s".format(colored(time.time()-startTime,'red'))
yValues = np.ravel(yValues)
from IPython import embed; embed()
#yValues = yValues.T
X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.2)#, random_state=42) #explain hypervariables
print (X_train.shape, y_train.shape)
#---------------------------------------------------------------------------------------------------------
'''
#est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
est2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,random_state=0)).fit(X_train, y_train)
mean_squared_error(y_test, est2.predict(X_test))  
clf = MultiOutputRegressor(neighbors.KNeighborsRegressor()).fit(X_train,y_train)

print (clf)

y_pred = clf.predict(X_test)
print(metrics.r2_score(y_test,y_pred,multioutput='raw_values'))
from IPython import embed; embed()

#---------------------------------------------------------------------------------------------------------
#regression
import time

from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

# Training classifiers
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=100)
reg2 = RandomForestRegressor(random_state=1, n_estimators=100)
reg3 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
ereg = MultiOutputRegressor(VotingRegressor([('gb', reg1), ('rf', reg2), ('lr', reg3)]))
print('start training')
#estimation, lower_bound, upper_bound
ereg.fit(X_train,y_train)
y_pred = ereg.predict(X_test)
'''
#---------------------------------------------------------------------------------------------------------
#classifiers

from sklearn.multioutput import MultiOutputClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Training classifiers
print "pre-classification"
from IPython import embed; embed()
clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10))
eclf = clf1.fit(X_train, y_train)
y_pred = eclf.predict(X_test)
metrics.confusion_matrix(y_test, y_pred, labels=range(16))

#clf2 = AdaBoostClassifier(KNeighborsClassifier(n_neighbors=10))
clf2 = AdaBoostClassifier(SVC(gamma='scale', kernel='rbf', probability=True))

clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=1000,n_jobs=-1))
bclf = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)
#clf4 = RandomForestClassifier(n_estimators=50, random_state=1)
#clf5 = GaussianNB()
#eclf = MultiOutputClassifier(VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1, 1, 1]))
#eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3), ('rf', clf4), ('gnb', clf5)], voting='soft')
#eclf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=600,learning_rate=1)
algList = [clf1]
print('start training')
try:
	fitted = []
	predictions = []
	for idx,alg in enumerate(algList):
		startTime = time.time()
		print "Starting training for clf{}".format(idx)
		fitted.append(alg.fit(X_train, y_train))
		print "Finished training: {:.4}s".format(colored(time.time()-startTime,'red'))

		startTime = time.time()
		print "Starting predictions for clf{}".format(idx)
		predictions.append(alg.predict(X_test))
		print "Finished predictions: {:.4}s".format(colored(time.time()-startTime,'red'))
		'''
		startTime = time.time()
		print "Starting cross_validate for clf{}".format(idx)
		scores = cross_val_score(fitted[idx], xValues, yValues, cv=5)
		print "Finsihed cross_validate: {:.4}s".format(time.time()-startTime)
		print("%s Accuracy: %0.2f (+/- %0.2f)" % (str(idx),scores.mean(), scores.std() * 2))
		
		metrics.confusion_matrix(y_test.T[0], y_pred.T[0], labels=range(16)
		print("accuracy_score:", accuracy_score(predictions[idx], y_test))
		print(metrics.confusion_matrix(y_test.T[0], predictions[idx].T[0], labels=range(16)))
		print(metrics.confusion_matrix(y_test, predictions[idx], labels=range(16)))
		print("Scenario accuracy: "+str(np.sum(y_test==predictions[idx])/len(y_test)*100)+"%")
		print("Scenario accuracy: "+str(np.sum(y_test.T[0]==predictions[idx].T[0])/len(y_test.T[0])*100)+"%")
		'''
		from IPython import embed; embed()
except Exception as e:
	print "error occured: {}".format(e)
	from IPython import embed; embed()
from IPython import embed; embed()
