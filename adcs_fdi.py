from numpy import *
import os
import csv
import sklearn
import sklearn.tree
import sklearn.neighbors
import sklearn.linear_model
from sklearn import preprocessing
from sklearn import utils

outputFolder = "output"

trainRatio = 0.8
#testRatio = 1.0 - trainRatio <-- assumed

outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = sorted(outputDataset, key=lambda i: float(i.split("_")[0]))

inputDict = {}
outputDict = {}
for data in outputDataset:
	dataPointNum = int(data.split('_')[0])
	inputDict[dataPointNum] = array([float(i) for i in data.replace('.csv','').split('_')[1:]])
	outputDict[dataPointNum] = genfromtxt('output/'+data, delimiter=',',skip_header=1)

trainNum = round(len(inputDict)*trainRatio)

inputTrain = dict(list(inputDict.items())[:trainNum])
outputTrain = dict(list(outputDict.items())[:trainNum])

inputTest = dict(list(inputDict.items())[trainNum:])
outputTest = dict(list(outputDict.items())[trainNum:])

from IPython import embed; embed()

model1 = sklearn.tree.DecisionTreeClassifier()
model2 = sklearn.neighbors.KNeighborsClassifier()
model3 = sklearn.linear_model.LogisticRegression()

model1.fit(list(outputTrain.values())[],list(inputTrain.values())[0])
model2.fit(list(outputTrain.values()),list(inputTrain.values()))
model3.fit(list(outputTrain.values()),list(inputTrain.values()))

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3