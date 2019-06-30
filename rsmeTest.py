import numpy as np
import os
import matplotlib.pyplot as plt
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

outputFolder = "output_625"

listCutFactor = 3
outputDataset = [file for file in os.listdir(outputFolder) if file.endswith(".csv")]
outputDataset = outputDataset[::listCutFactor]

timeframe = 10
csvTimestep = 0.1

inputDict = {}
outputDict = {}

xNames = ['q1_faulty','q2_faulty','q3_faulty','omega1_faulty','omega2_faulty','omega3_faulty']
xNamesNominal = ['q1_healthy','q2_healthy','q3_healthy','omega1_healthy','omega2_healthy','omega3_healthy']
#xNames = ['time','q1_faulty','q2_faulty','q3_faulty','q4_faulty','omega1_faulty','omega2_faulty','omega3_faulty','wheel1_i_faulty','wheel1_w_faulty','wheel2_i_faulty','wheel2_w_faulty','wheel3_i_faulty','wheel3_w_faulty',\
#	'wheel4_i_faulty','wheel4_w_faulty']

#xValues = None
#yValues = None
xValues = []
yValues = []

print ("started data handling")
startTime = time.time()
def sav(y,n=51,d=3):
	return savgol_filter(y,n,d)

for data in outputDataset:
	dataPointNum = int(data.split('_')[0])
	scenarioNum = int(data.split('_')[1])
	#if scenarioNum > 4:
	#	continue
	inputData = np.array([float(i) for i in data.replace('.csv','').split('_')[1:]])

	inceptionTime = int(inputData[3]/csvTimestep)
	timeframe = int(inputData[5]) if int(inputData[5]) >= 2 else 1
	endTime = int((inputData[3]+timeframe)/csvTimestep)
	#targetTime = 30
	#deviation = 3
	#if int(inputData[3]) > targetTime+deviation or int(inputData[3]) < targetTime-deviation:
	#	continue

	outputData = pd.read_csv(outputFolder+"/"+data)

	#set rsme values  transformer.fit_transform(as features
	inputValues = np.take(inputData,[0]).astype(int)

	#int(inputData[5])
	#faulty = normalize(outputData[xNames].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	#nominal = normalize(outputData[xNamesNominal].values[inceptionTime:inceptionTime+int(timeframe/csvTimestep)])
	faulty = (outputData[xNames].values)[inceptionTime:endTime]
	nominal =  (outputData[xNamesNominal].values)[inceptionTime:endTime]
	#outputValues = [np.sqrt(mean_squared_error(nominal.T[i],faulty.T[i])) for i in range(len(xNames))] 	#.flatten()
	outputValuesQ = []
	outputValuesW = []
	for i in range(len(xNames)):
		try:
			n = len(nominal.T[i])-1 if len(nominal.T[i]) % 2 == 0 else len(nominal.T[i])
			nom = sav(nominal.T[i],n=n)
			fal = sav(faulty.T[i],n=n)
			#nom,fal = normalize([nom,fal])
			residual = nom-fal
			#from IPython import embed; embed()
			outputValuesQ.append(np.sqrt(mean_squared_error(nom,fal)))
			outputValuesQ.append(np.max(residual))
			outputValuesQ.append(np.std(residual))
			outputValuesQ.append(np.min(residual))
			outputValuesQ.append(np.mean(residual))
			outputValuesQ.append(np.var(residual))
			outputValuesQ.append(mean_absolute_error(nom,fal))
		except Exception as e: 
			print "Error!:", e,"\n"
			from IPython import embed; embed()
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

#yValues = np.ravel(yValues)
yValues = np.array(yValues)
xValues = np.array(xValues)
#from IPython import embed; embed()
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



#pca = PCA(n_components=3)
#pcatrain = pca.fit_transform(X_train)
#pcatest = pca.fit_transform(X_test)
#v = #KMeans(n_clusters=5,random_state=123)
n_components = 4
#v = GaussianMixture(n_components=n_components, covariance_type='full')
#y_pred_v = v.fit_predict(X_train)
#y_pred_v = v.fit_predict(pcatst)
#clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0))
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print "RF accuracy_score: {}".format(accuracy_score(y_pred,y_test))
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