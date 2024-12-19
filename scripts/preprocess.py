from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.utils import separate_features_label
from src.preprocessing import normalize_features, label_encoding, smote
from src.preprocessing import remove_outliers
from zipfile import ZipFile

def preprocess_data():
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
    X_train_scaled_2021_22, X_test_scaled_2021_22 = normalize_features(X_resampled_21_22, X_resampled_23)
    X_train_scaled_2021_23, X_test_scaled_2021_23 = normalize_features(X_resampled_21_23, X_resampled_22)
    X_train_scaled_2022_23, X_test_scaled_2022_23 = normalize_features(X_resampled_22_23, X_resampled_21)
    y_train_encoded_2021_22, y_test_encoded_2021_22 = label_encoding(y_resampled_21_22, y_resampled_21_22)
    y_train_encoded_2021_23, y_test_encoded_2021_23 = label_encoding(y_resampled_21_23, y_resampled_21_23)
    y_train_encoded_2022_23, y_test_encoded_2022_23 = label_encoding(y_resampled_22_23, y_resampled_22_23)
    
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
    
    return X_train_scaled_2021_22,X_test_scaled_2021_22, y_train_encoded_2021_22,y_test_encoded_2021_22,X_train_scaled_2021_23, X_test_scaled_2021_23, y_test_encoded_2021_23,y_train_encoded_2021_23, X_test_scaled_2022_23, y_test_encoded_2022_23,X_train_scaled_2022_23,y_train_encoded_2022_23, X_scaled_21, y_encoded_21, X_scaled_22, y_encoded_22, X_scaled_23, y_encoded_23
