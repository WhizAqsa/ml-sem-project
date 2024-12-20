from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from src.plots import plot_dbscan_clusters_without_pca,plot_dbscan_clusters_with_pca, plot_class_distribution_in_clusters,visualize_confusion_matrix,plot_k_distance
from src.evaluation import calculate_cluster_purity, get_confusion_matrix
from sklearn.metrics import silhouette_score
import pandas as pd
from zipfile import ZipFile
from src.preprocessing import smote
from src.utils import separate_features_label
from sklearn.preprocessing import StandardScaler,LabelEncoder
import numpy as np

#import warnings
#warnings.filterwarnings("ignore", message=".*n_init.*", category=FutureWarning)

def run_dbscan_model():
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
    
    
    # DBSCAN Clustering
    print("--------------------------------------------------------")
    print("Perform dbscan Clustering on 2021-22 (crop,rice) dataset")
    print("--------------------------------------------------------")
    
    
    # Find the optimal eps using the k-distance plot
    plot_k_distance(X_scaled_21, k=4)  # Choose k based on dataset size (e.g., k=4 or 5)

    # Define potential hyperparameters
    eps_values = np.arange(0.1, 2.0, 0.1)  # Adjust range as per k-distance plot
    min_samples_values = [5, 10, 15]  # Adjust based on dataset size

    optimal_eps = None
    best_min_samples = None
    best_silhouette = -1

    # Grid search for optimal eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled_21)
            
            # Check if valid clusters are formed
            num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
            if num_clusters > 1:
                # Calculate silhouette score
                silhouette_avg = silhouette_score(X_scaled_21, clusters)
                print(f"eps: {eps}, min_samples: {min_samples}, Silhouette: {silhouette_avg:.2f}, Clusters: {num_clusters}")
                
                # Update best parameters
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    optimal_eps = eps
                    best_min_samples = min_samples

    # Train DBSCAN with best parameters
    print(f"Best eps: {optimal_eps}, Best min_samples: {best_min_samples}, Best Silhouette: {best_silhouette:.2f}")
    dbscan = DBSCAN(eps=optimal_eps, min_samples=best_min_samples)
    clusters_21 = dbscan.fit_predict(X_scaled_21)

    # Check final cluster count
    num_clusters = len(np.unique(clusters_21)) - (1 if -1 in clusters_21 else 0)
    print(f"Final number of clusters: {num_clusters}")

    # Visualize clusters
    plot_class_distribution_in_clusters(y_encoded_21, clusters_21, "DBSCAN")
    # Visualize clusters with PCA
    print("Visualizing clusters with PCA...")
    plot_dbscan_clusters_with_pca(X_scaled_21,eps=optimal_eps,min_samples=best_min_samples,n_components=2)
    
    # Visualize clusters without PCA
    print("Visualizing clusters without PCA...")
    plot_dbscan_clusters_without_pca(X_scaled_21,eps=optimal_eps,min_samples=best_min_samples)
    
    # Cluster purity
    purity = calculate_cluster_purity(y_encoded_21, clusters_21)
    print(f"Cluster purity: {purity:.2f}")
    
    # Confusion matrix (ignore noise points, where cluster == -1)
    clusters_21_filtered = clusters_21[clusters_21 != -1]
    y_encoded_21_filtered = y_encoded_21[clusters_21 != -1]
    print(f"Confusion matrix:\n{get_confusion_matrix(y_encoded_21_filtered, clusters_21_filtered)}")
    
    # Visualize the confusion matrix
    print("Visualizing the confusion matrix...")
    visualize_confusion_matrix(get_confusion_matrix(y_encoded_21_filtered, clusters_21_filtered))


if __name__ == "__main__":
    run_dbscan_model()