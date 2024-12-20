from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from src.plots import plot_gmm_clusters_without_pca, plot_gmm_clusters_with_pca, plot_class_distribution_in_clusters, visualize_confusion_matrix, plot_k_distance
from src.evaluation import calculate_cluster_purity, get_confusion_matrix
from sklearn.metrics import silhouette_score
import pandas as pd
from zipfile import ZipFile
from src.preprocessing import smote
from src.utils import separate_features_label
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def run_gmm_model():
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

    # GMM Clustering
    print("--------------------------------------------------------")
    print("Perform GMM Clustering on 2021-22 (crop,rice) dataset")
    print("--------------------------------------------------------")

    # Define potential hyperparameters
    n_components_values = range(2, 10)  # Adjust range based on dataset

    best_n_components = None
    best_silhouette = -1

    # Grid search for optimal number of components
    for n_components in n_components_values:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        clusters = gmm.fit_predict(X_scaled_21)

        # Check if valid clusters are formed
        num_clusters = len(np.unique(clusters))
        if num_clusters > 1:
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_scaled_21, clusters)
            print(f"n_components: {n_components}, Silhouette: {silhouette_avg:.2f}, Clusters: {num_clusters}")

            # Update best parameters
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_components = n_components

    # Handle edge case where no valid clusters are formed
    if best_n_components is None:
        print("No valid clusters formed. Consider adjusting the range of n_components.")
    else:
        print(f"Best n_components: {best_n_components}, Best Silhouette: {best_silhouette:.2f}")

        # Train GMM with best number of components
        gmm = GaussianMixture(n_components=best_n_components, random_state=42)
        clusters_21 = gmm.fit_predict(X_scaled_21)

        # Visualize results
        print("Visualizing clusters...")
        plot_class_distribution_in_clusters(y_encoded_21, clusters_21, "GMM")
        plot_gmm_clusters_with_pca(X_scaled_21, n_components=best_n_components, n_features=2)

        # Visualize clusters without PCA
        print("Visualizing clusters without PCA...")
        plot_gmm_clusters_without_pca(X_scaled_21, n_components=best_n_components)

        # Cluster purity
        purity = calculate_cluster_purity(y_encoded_21, clusters_21)
        print(f"Cluster purity: {purity:.2f}")

        # Confusion matrix
        print(f"Confusion matrix:\n{get_confusion_matrix(y_encoded_21, clusters_21)}")

        # Visualize the confusion matrix
        print("Visualizing the confusion matrix...")
        visualize_confusion_matrix(get_confusion_matrix(y_encoded_21, clusters_21))


if __name__ == "__main__":
    run_gmm_model()
