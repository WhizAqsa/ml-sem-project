from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from src.plots import (
    plot_k_values_with_elbow_method,
    plot_kmeans_clusters_without_pca,
    plot_kmeans_clusters_with_pca,
    visualize_confusion_matrix,
    plot_kmeans_class_distribution_in_clusters,
)
from src.evaluation import calculate_cluster_purity, get_confusion_matrix
import pandas as pd
from zipfile import ZipFile
from src.preprocessing import smote
from src.utils import separate_features_label
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore", message=".*n_init.*", category=FutureWarning)


def run_kmeans_model():
    # unzip the data file
    file_name = "data/raw/Crop-dataset.zip"
    with ZipFile(file_name, "r") as zip:
        print("Extracting all the files now...")
        zip.extractall("data/raw/")
        print("Done extracting the files")

    # load the datasets
    print("Loading the datasets...")
    cotton_dataset_2021 = pd.read_csv("data/raw/Crop-dataset/Cotton/cotton2021.csv")
    cotton_dataset_2022 = pd.read_csv("data/raw/Crop-dataset/Cotton/cotton2022.csv")
    cotton_dataset_2023 = pd.read_csv("data/raw/Crop-dataset/Cotton/cotton2023.csv")
    rice_dataset_2021 = pd.read_csv("data/raw/Crop-dataset/Rice/rice2021.csv")
    rice_dataset_2022 = pd.read_csv("data/raw/Crop-dataset/Rice/rice2022.csv")
    rice_dataset_2023 = pd.read_csv("data/raw/Crop-dataset/Rice/rice2023.csv")

    # add the label column
    cotton_dataset_2021["label"] = "cotton"
    cotton_dataset_2022["label"] = "cotton"
    cotton_dataset_2023["label"] = "cotton"
    rice_dataset_2021["label"] = "rice"
    rice_dataset_2022["label"] = "rice"
    rice_dataset_2023["label"] = "rice"

    # merge the cotton and rice datasets
    cotton_rice_2021 = pd.concat(
        [cotton_dataset_2021.copy(), rice_dataset_2021.copy()], ignore_index=True
    )
    cotton_rice_2022 = pd.concat(
        [cotton_dataset_2022.copy(), rice_dataset_2022.copy()], ignore_index=True
    )
    cotton_rice_2023 = pd.concat(
        [cotton_dataset_2023.copy(), rice_dataset_2023.copy()], ignore_index=True
    )

    # combine all datasets
    cotton_rice_all = pd.concat(
        [cotton_rice_2021.copy(), cotton_rice_2022.copy(), cotton_rice_2023.copy()],
        ignore_index=True,
    )

    # separate features and label
    print("Separating features and label...")
    X_all, y_all = separate_features_label(cotton_rice_all)
    print(
        f"Before oversampling:\nX_all shape: {X_all.shape}, y_all shape: {y_all.shape}"
    )

    # handle class imbalance
    print("Handling class imbalance...")
    X_resampled_all, y_resampled_all = smote(X_all, y_all)

    print(
        f"After oversampling:\nX_resampled_all shape: {X_resampled_all.shape}, y_resampled_all shape: {y_resampled_all.shape}"
    )

    # normalize the features
    print("Normalizing the features...")
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_resampled_all)

    # encode the labels
    encoder = LabelEncoder()
    y_encoded_all = encoder.fit_transform(y_resampled_all)

    # KMeans Clustering
    print("--------------------------------------------------------")
    print("Perform KMeans Clustering on combined dataset (2021, 2022, 2023)")
    print("--------------------------------------------------------")

    # determine the optimal no of k
    # using elbow method
    print("Visualizing different `k` values...")
    plot_k_values_with_elbow_method(X_scaled_all)

    for k in range(2, 5):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=1)

        # Predict the cluster labels
        print(f"Predicting clusters for k={k}...")
        clusters = kmeans.fit_predict(X_scaled_all)
        print(f"Clusters: {clusters}")

        # Visualize class distribution in clusters for different k values
        print(f"Visualizing class distribution in clusters for k={k}...")
        plot_kmeans_class_distribution_in_clusters(y_encoded_all, clusters, k)

        # Show clusters with PCA
        print("Visualizing clusters with PCA...")
        plot_kmeans_clusters_with_pca(X_scaled_all, clusters, k)

        # Show clusters without PCA
        print("Visualizing clusters without PCA...")
        plot_kmeans_clusters_without_pca(X_scaled_all, clusters, k)

        # cluster purity
        purity = calculate_cluster_purity(y_encoded_all, clusters)
        print(f"Cluster purity for K-means: {purity:.2f}")

        # confusion matrix
        print(f"Confusion matrix:\n{get_confusion_matrix(y_encoded_all, clusters)}")

        # visualize the confusion matrix
        print("Visualize the confusion matrix...")
        visualize_confusion_matrix(get_confusion_matrix(y_encoded_all, clusters))


if __name__ == "__main__":
    run_kmeans_model()
