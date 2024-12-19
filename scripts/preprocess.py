from sklearn.model_selection import train_test_split
import pandas as pd
from src.preprocessing import normalize_features, label_encoding, smote

def preprocess_data(X,y):
    #---------------------------------------------------------------------#
    # handle class imbalance
    X_resampled, y_resampled = smote(X, y)

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.2, random_state=42)    

    # normalize the features
    X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

    # label encoding of label class
    y_train_encoded, y_test_encoded = label_encoding(y_train, y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
