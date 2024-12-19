import pandas as pd
from src.utils import fit_and_get_predictions,separate_features_label, get_best_params_grid_search
from src.preprocessing import smote, remove_outliers
from scripts.preprocess import preprocess_data
from sklearn.svm import SVC
from src.evaluation import get_eval_metrics
from src.plots import visualize_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile
import sys


def run_svm_model():
    # unzip the data file
    file_name = "data/raw/Crop-dataset.zip"
    with ZipFile(file_name, 'r') as zip:
        print('Extracting all the files now...')
        zip.extractall("data/raw/")
        print('Done extracting the files')
        
    # load the datasets
    print("Loading the datasets...")
    cotton_dataset_2021 = pd.read_csv('data/raw/Crop-dataset/Cotton/cotton2021.csv')
    cotton_dataset_2022 = pd.read_csv('data/raw/Crop-dataset/Cotton/cotton2022.csv')
    cotton_dataset_2023 = pd.read_csv('data/raw/Crop-dataset/Cotton/cotton2023.csv')
    rice_dataset_2021 = pd.read_csv('data/raw/Crop-dataset/Rice/rice2021.csv')
    rice_dataset_2022 = pd.read_csv('data/raw/Crop-dataset/Rice/rice2022.csv')
    rice_dataset_2023 = pd.read_csv('data/raw/Crop-dataset/Rice/rice2023.csv')
    
    # add the label column
    cotton_dataset_2021['label'] = 'cotton'
    cotton_dataset_2022['label'] = 'cotton'
    cotton_dataset_2023['label'] = 'cotton'
    rice_dataset_2021['label'] = 'rice'
    rice_dataset_2022['label'] = 'rice'
    rice_dataset_2023['label'] = 'rice'
    
    # merge the cotton and rice datasets
    cotton_rice_2021 = pd.concat([cotton_dataset_2021.copy(), rice_dataset_2021.copy()], ignore_index=True)
    cotton_rice_2022 = pd.concat([cotton_dataset_2022.copy(), rice_dataset_2022.copy()], ignore_index=True)
    cotton_rice_2023 = pd.concat([cotton_dataset_2023.copy(), rice_dataset_2023.copy()], ignore_index=True)
    
    # create the 3 splits
    print("Creating the 3 splits...")
    dataset_2021_22 = pd.concat([cotton_rice_2021.copy(), cotton_rice_2022.copy()], ignore_index=True)
    dataset_2021_23 = pd.concat([cotton_rice_2021.copy(), cotton_rice_2023.copy()], ignore_index=True)
    dataset_2022_23 = pd.concat([cotton_rice_2022.copy(), cotton_rice_2023.copy()], ignore_index=True)
    
    # separate features and label
    print("Separating features and label...")
    X_2021_22, y_2021_22 = separate_features_label(dataset_2021_22)
    X_2021_23, y_2021_23 = separate_features_label(dataset_2021_23)
    X_2022_23, y_2022_23 = separate_features_label(dataset_2022_23)
    X_2021, y_2021 = separate_features_label(cotton_rice_2021)
    X_2022, y_2022 = separate_features_label(cotton_rice_2022)
    X_2023, y_2023 = separate_features_label(cotton_rice_2023)
    
    # preprocess the data
    print("Preprocessing the data...")
    
    # handle class imbalance
    print("Handling class imbalance...")
    X_resampled_21_22, y_resampled_21_22 = smote(X_2021_22, y_2021_22)
    X_resampled_21_23, y_resampled_21_23 = smote(X_2021_23, y_2021_23)
    X_resampled_22_23, y_resampled_22_23 = smote(X_2022_23, y_2022_23)
    X_resampled_21, y_resampled_21 = smote(X_2021, y_2021)
    X_resampled_22, y_resampled_22 = smote(X_2022, y_2022)
    X_resampled_23, y_resampled_23 = smote(X_2023, y_2023)
    
    # remove the outliers
    print("Removing outliers...")
    X_resampled_21_22, y_resampled_21_22 = remove_outliers(X_resampled_21_22, y_resampled_21_22) 
    X_resampled_22_23, y_resampled_22_23 = remove_outliers(X_resampled_22_23, y_resampled_22_23)
    X_resampled_21_23, y_resampled_21_23 = remove_outliers(X_resampled_21_23, y_resampled_21_23)
    X_resampled_21, y_resampled_21 = remove_outliers(X_resampled_21, y_resampled_21)
    X_resampled_22, y_resampled_22 = remove_outliers(X_resampled_22, y_resampled_22)
    X_resampled_23, y_resampled_23 = remove_outliers(X_resampled_23, y_resampled_23) 
    
    # normalize the features
    print("Normalizing the features...")
    X_train_scaled_2021_22, X_test_scaled_2021_22, y_train_encoded_2021_22, y_test_encoded_2021_22 = preprocess_data(X_resampled_21_22, y_resampled_21_22,X_resampled_23, y_resampled_23)
    X_train_scaled_2021_23, X_test_scaled_2021_23, y_train_encoded_2021_23, y_test_encoded_2021_23 = preprocess_data(X_resampled_21_23, y_resampled_21_23,X_resampled_22, y_resampled_22)
    X_train_scaled_2022_23, X_test_scaled_2022_23, y_train_encoded_2022_23, y_test_encoded_2022_23 = preprocess_data(X_resampled_22_23, y_resampled_22_23,X_resampled_21, y_resampled_21)
    
    # scale the test datasets
    scaler = StandardScaler()
    X_scaled_21 = scaler.fit_transform(X_resampled_21)
    X_scaled_22 = scaler.transform(X_resampled_22)
    X_scaled_23 = scaler.transform(X_resampled_23)
    
    # encode the test datasets  
    encoder = LabelEncoder()
    y_encoded_21 = encoder.fit_transform(y_resampled_21)
    y_encoded_22 = encoder.transform(y_resampled_22)
    y_encoded_23 = encoder.transform(y_resampled_23)
    
    # run the default 
    print("Running default SVM model...")
    model = SVC(random_state=42)
    y_pred_rf = fit_and_get_predictions(model, X_train_scaled_2021_22, y_train_encoded_2021_22, X_scaled_23)
    
    # print the classification report
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_rf)
    print(f"Accuracy score for default SVM model: {score}")
    print(f"Classification report for default SVM model:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix...")
    visualize_confusion_matrix(cm)
    
    # grid search
    print("SVM Grid Search")
    params_grid_svm = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    best_params_svm = get_best_params_grid_search(
    SVC(random_state=42),
    X_train_scaled_2021_22,
    y_train_encoded_2021_22,
    params_grid_svm
    )
    print(f"SVM Best params: {best_params_svm}")
    best_svm_model = SVC(
    random_state=42,
    **best_params_svm
    )
    
    print("Running the best SVM model...")
    y_pred_svm_best = fit_and_get_predictions(
    best_svm_model,
    X_train_scaled_2021_22, 
    y_train_encoded_2021_22,
    X_scaled_23
    )
    
    # print the classification report for the best model
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_svm_best)
    print(f"Accuracy score for the best SVM model: {score}")
    print(f"Classification report for the best SVM model:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix after applying grid search...")
    visualize_confusion_matrix(cm)
    

if __name__ == "__main__":
    run_svm_model()