#!/usr/bin/env python
# coding: utf-8

# # Multiclass Example

# This example show shows how to use `tsfresh` to extract and select useful features from timeseries in a multiclass classification example.
# 
# We use an example dataset of human activity recognition for this.
# The dataset consists of timeseries for 7352 accelerometer readings. 
# Each reading represents an accelerometer reading for 2.56 sec at 50hz (for a total of 128 samples per reading). Furthermore, each reading corresponds one of six activities (walking, walking upstairs, walking downstairs, sitting, standing and laying).
# 
# For more information go to https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
# 
# This notebook follows the example in [the first notebook](./01%20Feature%20Extraction%20and%20Selection.ipynb), so we will go quickly over the extraction and focus on the more interesting feature selection in this case.

# In[ ]:

import matplotlib.pylab as plt

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
from IPython import embed


# ## Load and visualize data

# In[ ]:


from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes

# fetch dataset from uci
download_har_dataset()


# In[ ]:


df = load_har_dataset()
df.head()


# In[ ]:


y = load_har_classes()


# The data is not in a typical time series format so far: 
# the columns are the time steps whereas each row is a measurement of a different person.
# 
# Therefore we bring it to a format where the time series of different persons are identified by an `id` and are order by time vertically.

# In[ ]:


df["id"] = df.index
df = df.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)


# In[ ]:


df.head()


# In[ ]:


plt.title('accelerometer reading')
plt.plot(df[df["id"] == 0].set_index("time").value)
plt.show()


# ## Extract Features

# In[ ]:


# only use the first 500 ids to speed up the processing
X = extract_features(df[df["id"] < 500], column_id="id", column_sort="time", impute_function=impute)


# In[ ]:


X.head()


# ## Train and evaluate classifier

# For later comparison, we train a decision tree on all features (without selection):

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y[:500], test_size=.2)


# In[ ]:


classifier_full = DecisionTreeClassifier()
classifier_full.fit(X_train, y_train)
print(classification_report(y_test, classifier_full.predict(X_test)))


# # Multiclass feature selection

# We will now select a subset of relevant features using the `tsfresh` select features method.
# However it only works for binary classification or regression tasks. 
# 
# For a 6 label multi classification we therefore split the selection problem into 6 binary one-versus all classification problems. 
# For each of them we can do a binary classification feature selection:

# In[ ]:


relevant_features = set()

for label in y.unique():
    y_train_binary = y_train == label
    X_train_filtered = select_features(X_train, y_train_binary)
    print("Number of relevant features for class {}: {}/{}".format(label, X_train_filtered.shape[1], X_train.shape[1]))
    relevant_features = relevant_features.union(set(X_train_filtered.columns))


# In[ ]:


len(relevant_features)


# we keep only those features that we selected above, for both the train and test set

# In[ ]:


X_train_filtered = X_train[list(relevant_features)]
X_test_filtered = X_test[list(relevant_features)]


# and train again:

# In[ ]:


classifier_selected = DecisionTreeClassifier()
classifier_selected.fit(X_train_filtered, y_train)
print(classification_report(y_test, classifier_selected.predict(X_test_filtered)))
embed()

# It worked! The precision improved by removing irrelevant features.
