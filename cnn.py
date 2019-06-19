

# Imports
import numpy as np
import os
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train, labels_train, list_ch_train = read_data(data_path="./data/", split="train") # train
X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test") # test

assert list_ch_train == list_ch_test, "Mistmatch in channels!"

# Normalize?
X_train, X_test = standardize(X_train, X_test)

a,b,c,d = train_test_split(X_train, labels_train, 
												stratify = labels_train, random_state = 123)

import time
import pandas as pd
outputFolder = "../adcs_fdi/output_625"

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
X_tr, X_vld, lab_tr, lab_vld = train_test_split(xValues, yValues, test_size=0.2, random_state=123) #explain hypervariables


y_tr = one_hot(lab_tr.astype(int),n_class=int(max(lab_tr))+1)
y_vld = one_hot(lab_vld.astype(int),n_class=int(max(lab_tr))+1)
y_test = one_hot(labels_test.astype(int),n_class=int(max(lab_tr))+1)

# Imports

import tensorflow as tf

batch_size = 600       # Batch size
seq_len = X_tr.shape[1]          # Number of steps
learning_rate = 0.0001
epochs = 100

n_classes = int(max(lab_tr))+1
n_channels = X_tr.shape[2]
from IPython import embed; embed()


graph = tf.Graph()

# Construct placeholders
with graph.as_default():
	inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
	labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
	keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
	learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
	# (batch, 128, 9) --> (batch, 64, 18)
	conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
	
	# (batch, 64, 18) --> (batch, 32, 36)
	conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
	
	# (batch, 32, 36) --> (batch, 16, 72)
	conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
	
	# (batch, 16, 72) --> (batch, 8, 144)
	conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
							 padding='same', activation = tf.nn.relu)
	max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

with graph.as_default():
	# Flatten and add dropout
	flat = tf.reshape(max_pool_4, (-1, (X_tr.shape[1]/(2**4))*(X_tr.shape[2]*(2**4))))
	flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
	
	# Predictions
	logits = tf.layers.dense(flat, n_classes)
	
	# Cost function and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
	optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
	
	# Accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

#if (os.path.exists('checkpoints-cnn') == False):
#	mkdir checkpoints-cnn

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