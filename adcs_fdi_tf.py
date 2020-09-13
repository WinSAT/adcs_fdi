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
from hpsklearn import HyperoptEstimator, standard_scaler, xgboost_classification, random_forest, decision_tree, any_classifier, min_max_scaler, gradient_boosting, any_preprocessing, extra_trees
from hyperopt import tpe
from sklearn.pipeline import Pipeline
from tsfresh.examples import load_robot_execution_failures
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import RelevantFeatureAugmenter
import pickle
import tsfel
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

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
	bar = '█' * filled_length + '-' * (bar_length - filled_length)

	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

#import xgboost as xgb
#
#import lightgbm as lgbm

#outputFolder = "output_625"
#outputFolder = "../adcs_fdi/output_300_constSeverity_csvs"
#outputFolder = "../adcs_fdi/output_adcs_fdi_inputs_5000_constSeverity_singleFaults"

outputFolder = "../adcs_fdi/output_1000_constSeverity_singleFaults_randPre10Inception_remainDuration"
#outputFolder = "../adcs_fdi/output_1000_constSeverity_singleFaults_randPre10Inception_10to20secDuration"
#outputFolder = "../adcs_fdi/output_1000_randomSeverity_singleFaults_5to55Inception_randRemainDuration"

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
parser.add_argument('--ntrain', type=str, help='X features dataset')
parser.add_argument('--ntest', type=str, help='Y features dataset')
parser.add_argument('--ytrain', type=str, help='X features dataset')
parser.add_argument('--ytest', type=str, help='Y features dataset')

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

#datasetLimit = 2000
#datasetLimiter = {k:0 for k in [0,1,2,3,4]}

if args.x and args.y:
	print ('Importing datasets - x: {}, y: {}'.format(args.x, args.y))
	#X_filtered = pandas.read_csv(args.x, index_col=0)
	data_Y = pandas.read_csv(args.y, index_col=0,squeeze=True) #pandas series
	data_X = pandas.read_csv(args.x, index_col=0)
	data_X = data_X.astype({'id': int})
	#print (X_filtered.info())
	#y = pandas.read_csv(args.y, index_col=0, header=None)
elif not args.x and not args.fx and not args.ntrain:
	try:
		print ('Starting Data Handling')
		#data_X = pandas.DataFrame([],columns=columnNames)
		data_X = []
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
			#if datasetLimiter[int(dataSetParamsDict['scenario'])] > datasetLimit:
			#	continue
			#else:
			#	datasetLimiter[int(dataSetParamsDict['scenario'])] += 1
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
			
			#residuals = faulty - nominal
			#datasetCutLength = residuals.shape[1]
			#preDataFrameResiduals = vstack([tile(dataSetIdCounter,datasetCutLength),arange(datasetCutLength),residuals]).T
			#preDataFrameResiduals = vstack([tile(dataSetIdCounter,datasetCutLength),arange(datasetCutLength),residuals]).T
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
			#data_Y.append(0)
			#data_X.append(pandas.DataFrame(nominal.T))
			
			#remove index with all zero values
			nominal = delete(nominal.T,0,0)
			faulty = delete(faulty.T,0,0)
			data_Y.append(int(dataSetParamsDict['scenario']))
			if int(dataSetParamsDict['scenario']) == 0:
				data_X.append(nominal)
			else:
				data_X.append(faulty)
			#data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrameResiduals,columns=columnNames)],ignore_index=True)
			#data_Y.append(int(dataSetParamsDict['scenario']))
			#data_X = pandas.concat([data_X,pandas.DataFrame(preDataFrameResiduals,columns=columnNames)],ignore_index=True)
			dataSetIdCounter += 1
			#data_Y.append(inputValues)
			#data handling
			print_progress(dataSetID, len(outputDataset), prefix = 'Data Handling:')

		#data_Y = pandas.Series(data_Y)
		saveTime = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
		#data_X.to_csv('X_df_{}.csv'.format(saveTime))
		#data_Y.to_csv('y_ds_{}.csv'.format(saveTime))
	except Exception as err:
		print ("Data handling error:", err)
		embed()

data_X = array(data_X)
data_Y = array(data_Y)
from utilities import *

X_train, X_test, labels_train, labels_test = train_test_split(data_X, data_Y, test_size=0.1, random_state=42)

# Normalize?
X_train, X_test = standardize(X_train, X_test)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, 
												stratify = labels_train, random_state = 123)

y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)


# Imports
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

batch_size = 500       # Batch size
seq_len = X_tr.shape[1]          # Number of steps
learning_rate = 0.001
epochs = 100

n_classes = len(np.unique(data_Y))
n_channels = X_tr.shape[2]
graph = tf.Graph()

# Construct placeholders
with graph.as_default():
	inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
	labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
	keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
	learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


with graph.as_default():
	# (batch, 128, 9) --> (batch, 64, 18)
	filterCounter = n_channels*2
	conv1 = tf.layers.conv1d(inputs=inputs_, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
	filterCounter *= 2
	# (batch, 64, 18) --> (batch, 32, 36)
	conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
	filterCounter *= 2
	# (batch, 32, 36) --> (batch, 16, 72)
	conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
	filterCounter *= 2
	# (batch, 16, 72) --> (batch, 8, 144)
	conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
	
	filterCounter *= 2
	conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')

	filterCounter *= 2
	conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=filterCounter, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same')


with graph.as_default():
	# Flatten and add dropout
	flat = tf.reshape(max_pool_6, (-1, int(max_pool_6.shape.dims[1])*int(max_pool_6.shape.dims[2])))
	flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
	
	# Predictions
	logits = tf.layers.dense(flat, n_classes)
	
	# Cost function and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
	optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
	
	# Accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

if (os.path.exists('checkpoints-cnn') == False):
	os.mkdir('checkpoints-cnn')

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
	saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
	sess.run(tf.global_variables_initializer())
	iteration = 1
   
	# Loop over epochs
	for e in range(epochs):
		
		# Loop over batches
		for x,y in get_batches(X_tr, y_tr, batch_size):
			
			# Feed dictionary
			feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
			
			# Loss
			loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
			train_acc.append(acc)
			train_loss.append(loss)
			
			# Print at each 5 iters
			if (iteration % 5 == 0):
				print("Epoch: {}/{}".format(e, epochs),
					  "Iteration: {:d}".format(iteration),
					  "Train loss: {:6f}".format(loss),
					  "Train acc: {:.6f}".format(acc))
			
			# Compute validation loss at every 10 iterations
			if (iteration%10 == 0):                
				val_acc_ = []
				val_loss_ = []
				
				for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
					# Feed
					feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
					
					# Loss
					loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                   
					val_acc_.append(acc_v)
					val_loss_.append(loss_v)
				
				# Print info
				print("Epoch: {}/{}".format(e, epochs),
					  "Iteration: {:d}".format(iteration),
					  "Validation loss: {:6f}".format(np.mean(val_loss_)),
					  "Validation acc: {:.6f}".format(np.mean(val_acc_)))
				
				# Store
				validation_acc.append(np.mean(val_acc_))
				validation_loss.append(np.mean(val_loss_))
			
			# Iterate 
			iteration += 1
	
	saver.save(sess,"checkpoints-cnn/har.ckpt")

# Plot training and test loss
try:
	t = np.arange(iteration-1)
	
	plt.figure(figsize = (6,6))
	plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
	plt.xlabel("iteration")
	plt.ylabel("Loss")
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()
	
	# Plot Accuracies
	plt.figure(figsize = (6,6))
	
	plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
	plt.xlabel("iteration")
	plt.ylabel("Accuray")
	plt.legend(['train', 'validation'], loc='upper right')
	plt.show()
	
except Exception as err:
	print ("Tensorflow error:",err)
	embed()

test_acc = []

with tf.Session(graph=graph) as sess:
	# Restore
	saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))
	
	for x_t, y_t in get_batches(X_test, y_test, batch_size):
		feed = {inputs_: x_t,
				labels_: y_t,
				keep_prob_: 1}
		
		batch_acc = sess.run(accuracy, feed_dict=feed)
		test_acc.append(batch_acc)
	print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

embed()