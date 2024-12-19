from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from src.plots import plot_k_values_with_elbow_method,plot_clusters_without_pca,plot_clusters_with_pca, plot_class_distribution_in_clusters,visualize_confusion_matrix
from src.evaluation import calculate_cluster_purity, get_confusion_matrix
import pandas as pd
from zipfile import ZipFile
from src.preprocessing import smote
from src.utils import separate_features_label
from sklearn.preprocessing import StandardScaler,LabelEncoder
import warnings
warnings.filterwarnings("ignore", message=".*n_init.*", category=FutureWarning)

def run_kmeans_model():
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

    # separate features and label
    print("Separating features and label...")
    X_2021, y_2021 = separate_features_label(cotton_rice_2021)
    X_2022, y_2022 = separate_features_label(cotton_rice_2022)
    X_2023, y_2023 = separate_features_label(cotton_rice_2023)
    print(f"Before oversampling:\nX_2021 shape: {X_2021.shape}, y_2021 shape: {y_2021.shape}")
    
    # handle class imbalance
    print("Handling class imbalance...")
    X_resampled_21, y_resampled_21 = smote(X_2021, y_2021)
    X_resampled_22, y_resampled_22 = smote(X_2022, y_2022)
    X_resampled_23, y_resampled_23 = smote(X_2023, y_2023)
    
    print(f"After oversampling:\nX_resampled_21 shape: {X_resampled_21.shape}, y_resampled_21 shape: {y_resampled_21.shape}")
    
    # normalize the features
    print("Normalizing the features...")
    scaler = StandardScaler()
    X_scaled_21 = scaler.fit_transform(X_resampled_21)
    X_scaled_22 = scaler.transform(X_resampled_22)
    X_scaled_23 = scaler.transform(X_resampled_23)
    
    # encode the test datasets  
    encoder = LabelEncoder()
    y_encoded_21 = encoder.fit_transform(y_resampled_21)
    y_encoded_22 = encoder.transform(y_resampled_22)
    y_encoded_23 = encoder.transform(y_resampled_23)
    
    
    #-------------------------1st Split----------------------------
    print("--------------------------------------------------------")
    print("Perform KMeans Clustering on 2021-22 (crop,rice) dataset")
    print("--------------------------------------------------------")
    
    # determine the optimal no of k
    # using elbow method
    print("Visualizing different `k` values...")
    plot_k_values_with_elbow_method(X_scaled_21)
        
    for k in range(2, 5):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1)
        
        # Predict the cluster labels
        print(f"Predicting clusters for k={k}...")
        clusters = kmeans.fit_predict(X_scaled_21)
        print(f"Clusters: {clusters}")
        
        # Visualize class distribution in clusters for different k values
        print(f"Visualizing class distribution in clusters for k={k}...")
        plot_class_distribution_in_clusters(y_encoded_21, clusters, k)
        
        # Show clusters with PCA
        print("Visualizing clusters with PCA...")
        plot_clusters_with_pca(X_scaled_21, clusters, k)
        
        # Show clusters without PCA
        print("Visualizing clusters without PCA...")
        plot_clusters_without_pca(X_scaled_21, clusters, k)
    
        # cluster purity
        purity = calculate_cluster_purity(y_encoded_21,clusters)
        print(f"Cluster purity for K-means: {purity:.2f}")

        # confusion matrix
        print(f"Confusion matrix:\n{get_confusion_matrix(y_encoded_21,clusters)}")
        
        # visalize the confusion matrix
        print("Visualize the confusion matrix...")
        visualize_confusion_matrix(get_confusion_matrix(y_encoded_21,clusters))



if __name__ == "__main__":
    run_kmeans_model()