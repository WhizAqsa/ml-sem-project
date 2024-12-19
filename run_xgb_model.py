import pandas as pd
from src.utils import fit_and_get_predictions,separate_features_label, get_best_params_grid_search
from src.preprocessing import smote, remove_outliers
from scripts.preprocess import preprocess_data
from xgboost import XGBClassifier
from src.evaluation import evaluate_model
from src.plots import visualize_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from zipfile import ZipFile
import sys


"""
1. Load all the dataset
    - you noob
    - merge rice and cotton of all three years
2. Create the 3 splits
    i.e. 21,22 and 23
         22,23 and 21
         21,23 and 22
3. Run default model on the 3 splits.
4. Do grid search on the whole dataset.
5. Perform cross validation.

all results should be properly logged.
all the plots should either be saved in a directory or be shown at runtime
all the results should also be logged to a text file where you can then show it to sir
"""


def run_xgb_model():s
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
    
    # save the preprocessed data
    print("Saving the preprocessed data...")

    # Combine and add labels for 2021-22
    combined_df_2021_22 = pd.concat([pd.DataFrame(X_train_scaled_2021_22), pd.DataFrame(X_test_scaled_2021_22)], ignore_index=True)
    combined_labels_2021_22 = pd.concat([pd.Series(y_train_encoded_2021_22), pd.Series(y_test_encoded_2021_22)], ignore_index=True)
    combined_df_2021_22['label'] = combined_labels_2021_22

    # Combine and add labels for 2021-23
    combined_df_2021_23 = pd.concat([pd.DataFrame(X_train_scaled_2021_23), pd.DataFrame(X_test_scaled_2021_23)], ignore_index=True)
    combined_labels_2021_23 = pd.concat([pd.Series(y_train_encoded_2021_23), pd.Series(y_test_encoded_2021_23)], ignore_index=True)
    combined_df_2021_23['label'] = combined_labels_2021_23

    # Combine and add labels for 2022-23
    combined_df_2022_23 = pd.concat([pd.DataFrame(X_train_scaled_2022_23), pd.DataFrame(X_test_scaled_2022_23)], ignore_index=True)
    combined_df_2021_22.to_csv('data/processed/cotton_rice_2021_22.csv', index=False)
    combined_df_2021_23.to_csv('data/processed/cotton_rice_2021_23.csv', index=False)
    combined_df_2022_23.to_csv('data/processed/cotton_rice_2022_23.csv', index=False)
      
    # run the default 
    print("Running default xgb model...")
    y_pred_xgb = fit_and_get_predictions(XGBClassifier(random_state=42), X_train_scaled_2021_22, y_train_encoded_2021_22, X_scaled_23)
    
    # print the classification report
    score, eval_report, cm = evaluate_model(y_encoded_23, y_pred_xgb)
    print(f"Classification report for default xgb model: {eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix...")
    visualize_confusion_matrix(cm)
    
    # grid search
    print("XGBoost Grid Search")
    params_grid_xgboost = {
    "max_depth": [3 ,5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "n_estimators": [100, 200, 300]
    }
    best_params_xgb = get_best_params_grid_search(
    XGBClassifier(random_state=42),
    X_train_scaled_2021_22,
    y_train_encoded_2021_22,
    params_grid_xgboost
    )
    print(f"XGBoost Best params: {best_params_xgb}")
    best_xgb_model = XGBClassifier(
    random_state=42,
    **best_params_xgb
    )
    
    print("Running the best xgb model...")
    y_pred_xgb_best = fit_and_get_predictions(
    best_xgb_model,
    X_train_scaled_2021_22, 
    y_train_encoded_2021_22,
    X_scaled_23
    )
    
    # print the classification report for the best model
    score, eval_report, cm = evaluate_model(y_encoded_23, y_pred_xgb)
    print(f"Classification report for the best xgb model: {eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix...")
    visualize_confusion_matrix(cm)
    

if __name__ == "__main__":
    run_xgb_model()