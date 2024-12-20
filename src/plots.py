from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_histogram(dataset):
  dataset.hist(bins=50, figsize=(20, 15))
  plt.show()
  
  
def plot_boxplot(dataset):
  figure = plt.figure(figsize=(12, 10))
  dataset.boxplot()
  plt.show()
  
  
def visualize_confusion_matrix(cm):
  plt.figure(figsize=(8, 4))
  sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Cotton', 'Rice'], yticklabels=['Cotton', 'Rice'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()
  
  
def plot_k_values_with_elbow_method(X):
  inertia = []
  for i in range(1, 11):
      kmeans = KMeans(n_clusters=i, init='k-means++',random_state=1)
      kmeans.fit(X)
      inertia.append(kmeans.inertia_)
      # print(f"Inertia value for {i} clusters is {kmeans.inertia_}")
  plt.plot(range(1, 11), inertia)
  plt.title('Elbow Method')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.show()
  
  
def analyze_silhouette_score(X):
  silhouette_scores = []
  for k in range(2, 11):
      kmeans = KMeans(n_clusters=k, random_state=1)
      clusters = kmeans.fit_predict(X)
      silhouette_scores.append(silhouette_score(X, clusters))
      print(f"Silhouette Score for {k} clusters is {silhouette_score(X, clusters)}")
  plt.plot(range(2, 11), silhouette_scores)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Silhouette Score')
  plt.show()
  
  
def plot_kmeans_clusters_with_pca(X,y,k):
  kmeans = KMeans(n_clusters=k, random_state=1)
  pca = PCA(n_components=0.90)
  pca_features = pca.fit_transform(X)
  # Fit KMeans on the PCA-transformed features
  new_labels = kmeans.fit_predict(pca_features)
  plt.scatter(pca_features[:, 0], pca_features[:, 1], c=new_labels, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('K-Means Clustering with PCA')
  plt.colorbar(label='Cluster Label')
  plt.show()
  
def plot_kmeans_clusters_without_pca(X, y, k):
    kmeans = KMeans(n_clusters=k, random_state=1)
    new_labels = kmeans.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=new_labels, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering without PCA')
    plt.colorbar(label='Cluster Label')
    plt.show()
  

def plot_sample_area_curve(sample_no,dataset):
    sample_data = dataset.iloc[sample_no][['NDVI01', 'NDVI02', 'NDVI03', 'NDVI04', 'NDVI05', 'NDVI06',
                                                'NDVI07', 'NDVI08', 'NDVI09', 'NDVI10', 'NDVI11', 'NDVI12']]
    label = dataset.iloc[sample_no].label

    # Convert the NDVI values to numeric using pd.to_numeric(), replacing non-numeric values with NaN
    sample_data_array = sample_data.apply(pd.to_numeric, errors='coerce').to_numpy()
    # print(sample_data_array)
    # Create x-axis values representing time steps
    time_steps = range(1, 13)  # 1 to 12 for NDVI01 to NDVI12

    # Create the plot
    plt.plot(time_steps, sample_data_array, marker='o', linestyle='-')
    plt.xlabel('Time Step')
    plt.ylabel('NDVI Value')
    plt.title(f"NDVI Time Series for Sample-{sample_no} with {label}")

    # Calculate AUC using NumPy's trapz function
    auc = np.trapz(sample_data_array, time_steps)
    # Fill the area under the curve
    plt.fill_between(time_steps, sample_data_array, alpha=0.3)  # alpha controls the transparency
    # Add AUC value to the plot
    plt.text(6, 0.6, f'AUC: {auc:.2f}', ha='center', va='center')
    plt.show()
  
  
# Function to visualize class distribution in clusters as bar chart
def plot_kmeans_class_distribution_in_clusters(y_true, clusters, k):
  cluster_labels = np.unique(clusters)
  class_labels = np.unique(y_true)
  class_distribution = {cl: [] for cl in class_labels}
  
  for cluster in cluster_labels:
      indices = np.where(clusters == cluster)[0]
      for cls in class_labels:
          count = np.sum(y_true[indices] == cls)
          class_distribution[cls].append(count)
  
  # Bar Chart for Class Distribution
  x = np.arange(len(cluster_labels))
  width = 0.2  # Width of bars
  fig, ax = plt.subplots(figsize=(10, 6))
  
  for i, cls in enumerate(class_labels):
      ax.bar(x + i * width, class_distribution[cls], width, label=f'Class {cls}')
  ax.set_xlabel('Clusters')
  ax.set_ylabel('Number of Samples')
  ax.set_title(f'Class Distribution in Clusters (k={k})')
  ax.set_xticks(x + width / 2)
  ax.set_xticklabels([f'Cluster {cl}' for cl in cluster_labels])
  ax.legend()
  plt.grid()
  plt.show()
  
  
def plot_class_distribution_in_clusters(y_true, clusters, algorithm_name="DBSCAN"):
  # Create a DataFrame to associate the cluster labels with true labels
  data = pd.DataFrame({"True Labels": y_true, "Cluster Labels": clusters})
  
  # Remove noise points (DBSCAN assigns noise points a label of -1)
  data = data[data["Cluster Labels"] != -1]
  
  # Plot the class distribution for each cluster
  plt.figure(figsize=(10, 6))
  sns.countplot(x="Cluster Labels", hue="True Labels", data=data)
  plt.title(f"Class Distribution in Clusters ({algorithm_name})")
  plt.xlabel("Cluster Labels")
  plt.ylabel("Count")
  plt.legend(title="True Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()
  plt.show()  
  
def plot_dbscan_clusters_with_pca(X, eps=0.5, min_samples=5, n_components=2):
  # Fit DBSCAN
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  # Apply PCA
  pca = PCA(n_components=n_components)
  X_pca = pca.fit_transform(X)

  clusters = dbscan.fit_predict(X_pca)
  
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('DBSCAN Clustering with PCA')
  plt.colorbar(label='Cluster Label')
  plt.show()
  
  
  
def plot_dbscan_clusters_without_pca(X, eps=0.5, min_samples=5):
  # Fit DBSCAN
  dbscan = DBSCAN(eps=eps, min_samples=min_samples)
  clusters = dbscan.fit_predict(X)
  
  # Plot the clusters
  plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('DBSCAN Clustering without PCA')
  plt.colorbar(label='Cluster Label')
  plt.show()


def plot_k_distance(X, k):
  neighbors = NearestNeighbors(n_neighbors=k)
  neighbors_fit = neighbors.fit(X)
  distances, _ = neighbors_fit.kneighbors(X)
  distances = np.sort(distances[:, k-1], axis=0)
  plt.plot(distances)
  plt.xlabel("Points sorted by distance")
  plt.ylabel(f"{k}-distance")
  plt.title(f"{k}-distance Plot")
  plt.show()
  
  
def plot_gmm_clusters_with_pca(X, n_components, n_features=2):
  gmm = GaussianMixture(n_components=n_components, random_state=42)
  pca = PCA(n_components=n_features)
  X_pca = pca.fit_transform(X)
  clusters = gmm.fit_predict(X_pca)
  
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('GMM Clustering with PCA')
  plt.colorbar(label='Cluster Label')
  plt.show()
  

def plot_gmm_clusters_without_pca(X, n_components):
  gmm = GaussianMixture(n_components=n_components, random_state=42)
  clusters = gmm.fit_predict(X)
  
  plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('GMM Clustering without PCA')
  plt.colorbar(label='Cluster Label')
  plt.show()
  
  
def plot_dendrogram(X, method='ward'):
    """
    Plots a dendrogram for hierarchical clustering.
    
    Parameters:
        X (array-like): The data for clustering.
        method (str): The linkage method ('ward', 'single', 'complete', 'average').
    """
    # Calculate the linkage matrix
    linked = linkage(X, method=method)
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, truncate_mode='level', p=10)
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()


def plot_hierarchical_class_distribution_in_clusters(labels, clusters, n_clusters):
  cluster_distribution = []
  unique_labels = np.unique(labels)
  
  # Compute class distribution for each cluster
  for cluster in range(n_clusters):
      cluster_indices = np.where(clusters == cluster)[0]
      label_counts = {label: np.sum(labels[cluster_indices] == label) for label in unique_labels}
      cluster_distribution.append(label_counts)
  
  # Plot the distribution
  fig, ax = plt.subplots(figsize=(10, 7))
  for label in unique_labels:
      ax.bar(
          range(n_clusters), 
          [dist.get(label, 0) for dist in cluster_distribution], 
          label=f"Class {label}"
      )
  
  plt.title("Class Distribution in Clusters")
  plt.xlabel("Cluster")
  plt.ylabel("Count")
  plt.legend(title="Classes")
  plt.show()
  
  
# def plot_hierarchical_clusters_with_pca(X, clusters, n_clusters):
#   hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
#   pca = PCA(n_components=2)
#   X_pca = pca.fit_transform(X)
#   clusters = hierarchical.fit_predict(X_pca)
#   plt.figure(figsize=(10, 7))
#   for cluster in range(n_clusters):
#       cluster_indices = np.where(clusters == cluster)[0]
#       plt.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1], label=f"Cluster {cluster}")
  
#   plt.title("Clusters with PCA")
#   plt.xlabel("Principal Component 1")
#   plt.ylabel("Principal Component 2")
#   plt.legend(title="Clusters")
#   plt.show()
  
# rewrite the function to plot for n_PCA components but visualize only the first two components
# n_PCA is the number of principal components to keep
# n_clusters is the number of clusters
def plot_hierarchical_clusters_with_pca(X, clusters, n_clusters, n_PCA):
  hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
  pca = PCA(n_components=n_PCA)
  X_pca = pca.fit_transform(X)
  clusters = hierarchical.fit_predict(X_pca)
  plt.figure(figsize=(10, 7))
  for cluster in range(n_clusters):
      cluster_indices = np.where(clusters == cluster)[0]
      plt.scatter(X_pca[cluster_indices, 0], X_pca[cluster_indices, 1], label=f"Cluster {cluster}")
  
  plt.title("Clusters with PCA")
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.legend(title="Clusters")
  plt.show()
  
  
def plot_hierarchical_clusters_without_pca(X, clusters, n_clusters):
  hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
  clusters = hierarchical.fit_predict(X)
  plt.figure(figsize=(10, 7))
  for cluster in range(n_clusters):
      cluster_indices = np.where(clusters == cluster)[0]
      plt.scatter(X[cluster_indices, 0], X[cluster_indices, 1], label=f"Cluster {cluster}")
  
  plt.title("Clusters without PCA")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.legend(title="Clusters")
  plt.show()