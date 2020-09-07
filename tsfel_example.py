import warnings
warnings.filterwarnings('ignore')
'''
!pip install https://github.com/fraunhoferportugal/tsfel/archive/v0.1.3.zip >/dev/null 2>&1
from sys import platform
if platform == "linux" or platform == "linux2":
    !wget http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip >/dev/null 2>&1
else:
    !pip install wget >/dev/null 2>&1
    import wget
    wget.download('http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip')
'''
import tsfel
import glob
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, accuracy_score

from IPython import embed

sns.set()

# Unzip dataset
zip_ref = zipfile.ZipFile("UCI HAR Dataset.zip", 'r')
zip_ref.extractall()
zip_ref.close()

def fill_missing_values(df):
    """ Handle eventual missing data. Strategy: replace with mean.
    
      Parameters
      ----------
      df pandas DataFrame
      Returns
      -------
        Data Frame without missing values.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

#@title Data Preparation
# Load train and test signals
train_files = glob.glob("UCI HAR Dataset/train/Inertial Signals/*.txt")
test_files = glob.glob("UCI HAR Dataset/test/Inertial Signals/*.txt")

strain = np.vstack([[np.loadtxt(fl).T] for fl in train_files if 'body_acc' not in fl]).T
stest = np.vstack([[np.loadtxt(fl).T] for fl in test_files if 'body_acc' not in fl]).T

columns_names_train = [fl.split('_')[-3] + fl.split('_')[-2] for fl in train_files if 'body_acc' not in fl]
columns_names_test  = [fl.split('_')[-3] + fl.split('_')[-2] for fl in test_files if 'body_acc' not in fl]

x_train_sig = [pd.DataFrame(strain[i], columns=columns_names_train) for i in range(len(strain))]
x_test_sig = [pd.DataFrame(stest[i], columns=columns_names_test) for i in range(len(stest))]
embed()

y_test = np.loadtxt('UCI HAR Dataset/test/y_test.txt')
y_train = np.loadtxt('UCI HAR Dataset/train/y_train.txt')
activity_labels = np.array(pd.read_csv('UCI HAR Dataset/activity_labels.txt', header=None, delimiter=' '))[:,1]

# dataset sampling frequency
fs = 100


window_number = 60
plot_axis = ['x', 'y', 'z']

plt.figure()
plt.subplot(2,1,1)
[plt.plot(np.arange(len(x_train_sig[window_number]))/fs, x_train_sig[window_number]['acc' + ax]) for ax in plot_axis]
plt.xlabel("time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Accelerometer Signal")
plt.legend(plot_axis)

plt.subplot(2,1,2)
[plt.plot(np.arange(len(x_train_sig[window_number]))/fs, x_train_sig[window_number]['gyro' + ax]) for ax in plot_axis]
plt.xlabel("time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Gyroscope Signal")
plt.legend(plot_axis)
plt.subplots_adjust(left=None, bottom=-0.3, right=None, top=1, wspace=None, hspace=0.5)

plt.show()

#@title Feature Extraction
googleSheet_name = "Features_dev"
# Extract excel info
cfg_file = tsfel.extract_sheet(googleSheet_name)

# Get features
X_train = tsfel.time_series_features_extractor(cfg_file, x_train_sig, fs=fs)
X_test = tsfel.time_series_features_extractor(cfg_file, x_test_sig, fs=fs)

# Handling eventual missing values from the feature extraction
X_train = fill_missing_values(X_train)
X_test = fill_missing_values(X_test)

# Highly correlated features are removed
corr_features = tsfel.correlated_features(X_train)
X_train.drop(corr_features, axis=1, inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)

# Remove low variance features
selector = VarianceThreshold()
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)

# Normalising Features
min_max_scaler = preprocessing.StandardScaler()
nX_train = min_max_scaler.fit_transform(X_train)
nX_test = min_max_scaler.transform(X_test)

classifier = RandomForestClassifier()
# Train the classifier
classifier.fit(nX_train, y_train.ravel())

# Predict test data
y_test_predict = classifier.predict(nX_test)

# Get the classification report
accuracy = accuracy_score(y_test, y_test_predict) * 100
print(classification_report(y_test, y_test_predict, target_names=activity_labels))
print("Accuracy: " + str(accuracy) + '%')

#@title Confusion Matrix
cm = confusion_matrix(y_test, y_test_predict)
df_cm = pd.DataFrame(cm, index=[i for i in activity_labels], columns=[i for i in activity_labels])
plt.figure()
ax= sns.heatmap(df_cm,  cbar=False, cmap="BuGn", annot=True, fmt="d")
plt.setp(ax.get_xticklabels(), rotation=45)

plt.ylabel('True label', fontweight='bold', fontsize = 18)
plt.xlabel('Predicted label', fontweight='bold', fontsize = 18)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()