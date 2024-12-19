from zipfile import ZipFile
from scripts.preprocess import preprocess_data
from scripts.unsupervised_models import run_unsupervised_models
import pandas as pd
from sklearn.svm import SVC
from src.utils import evaluate_model
from src.supervised import fit_and_predict_xgboost, random_forest_model, bagging_model, get_best_params_grid_search, fit_and_predict_svm


def separate_features_label(dataset):
    X = dataset.drop('label', axis=1)
    y = dataset['label']
    return X, y
        
    
# merege the two datasets
def merge_datasets(dataset1, dataset2):
    # adding a label column in both the datasets
    dataset1['label'] = 'cotton'
    dataset2['label'] = 'rice'
    # merge the two datasets
    merged_dataset = pd.concat([dataset1, dataset2], ignore_index=True)
    return merged_dataset

# Define the path where the file should be extracted
extract_to = "./data/raw/"

# unzip the data file
file_name = "./data/raw/Crop-dataset.zip"
with ZipFile(file_name, 'r') as zip:
  print('Extracting all the files now...')
  zip.extractall(extract_to)
  print('Done extracting the files')
  
  
# load the dataset
print("Loading the dataset")
cotton_2021 = pd.read_csv('./data/raw/Crop-dataset/Cotton/cotton2021.csv')
rice_2021 = pd.read_csv('./data/raw/Crop-dataset/Rice/rice2021.csv')
cotton_2021['label'] = 'cotton'
rice_2021['label'] = 'rice'

# merge the two datasets
print("Merge: 2021 Cotton, Rice")
cotton_rice_2021 = merge_datasets(cotton_2021, rice_2021)

# separate features and label
print("Separating features and label")
X, y = separate_features_label(cotton_rice_2021)

# preprocess the data
print("Preprocessing the data")
X_train_scaled, X_test_scaled,y_train_encoded,y_test_encoded = preprocess_data(X,y)

# combine X_train, X_test, y_train, y_test in a dataframe
print("Combining the data")
train = pd.DataFrame(X_train_scaled, columns=X.columns)
train['label'] = y_train_encoded
test = pd.DataFrame(X_test_scaled, columns=X.columns)
test['label'] = y_test_encoded

# combine train and test dataframes
combined_df = pd.concat([train, test], ignore_index=True)

# save the preprocessed data
print("Saving the preprocessed data")
combined_df.to_csv('./data/processed/cotton_rice_2021.csv', index=False)

# running supervised learning models
print("Running supervised learning models")
# XGBoost Classifier
print("Running XGBoost Classifier...")
y_pred_xgb = fit_and_predict_xgboost(X_train_scaled, y_train_encoded, X_test_scaled)
# Random Forest Classifier
print("Running Random Forest Classifier...")
y_pred_rf = random_forest_model(X_train_scaled, y_train_encoded, X_test_scaled)
# Bagging Classifier
print("Running Bagging Classifier...")
y_pred_bagging = bagging_model(X_train_scaled, y_train_encoded, X_test_scaled)
# Support Vector Machine Classifier
print("Running Grid Search CV to tune the hyperparameters...")
# Define a grid of hyperparameters
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['rbf']
}

# get the best hyperparameters
svm_best_params = get_best_params_grid_search(SVC(random_state=42),X_train_scaled, y_train_encoded,param_grid)
print(f"Best hyperparameters: {svm_best_params}")
print("Running Support Vector Machine Classifier...")
y_pred_svm = fit_and_predict_svm(X_train_scaled, y_train_encoded, X_test_scaled)


accuracy_score_xgb, classification_report_xgb, cm_xgb = evaluate_model(y_test_encoded, y_pred_xgb)
accuracy_score_rf, classification_report_rf, cm_rf = evaluate_model(y_test_encoded, y_pred_rf)
accuracy_score_bagging, classification_report_bagging, cm_bagging = evaluate_model(y_test_encoded, y_pred_bagging)
# run unsupervised learning models
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
X_df = pd.concat([X_train_scaled_df, X_test_scaled_df], ignore_index=True)
print("Running unsupervised learning models")
run_unsupervised_models(X_df)