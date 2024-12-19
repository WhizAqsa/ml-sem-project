import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats

# preprocess features
def normalize_features(X_train, X_test):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled


# label encoding of label class
def label_encoding(y_train, y_test):
  encoder = LabelEncoder()
  y_train = encoder.fit_transform(y_train)
  y_test = encoder.transform(y_test)
  return y_train, y_test


def smote(X,y):
  smote = SMOTE(random_state=42)
  X_resampled, y_resampled = smote.fit_resample(X, y)
  return X_resampled, y_resampled


def remove_outliers(X_resampled, y_resampled, columns):
  ndvi_columns = X_resampled[columns]
  z_scores = stats.zscore(ndvi_columns)
  threshold = 3
  mask = (np.abs(z_scores) < threshold).all(axis=1)
  X_no_outliers = X_resampled[mask]
  y_no_outliers = y_resampled[mask]
  return X_no_outliers, y_no_outliers