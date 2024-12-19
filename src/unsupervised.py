from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score


# determine the optimal no of k
# using elbow method
def elbow_method(X):
  inertia = []
  for i in range(1, 11):
      kmeans = KMeans(n_clusters=i, random_state=1)
      kmeans.fit(X)
      inertia.append(kmeans.inertia_)
      # print(f"Inertia value for {i} clusters is {kmeans.inertia_}")
  plt.plot(range(1, 11), inertia)
  plt.title('Elbow Method')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')
  plt.show()
  
  
def kmeans_model(X):
  k = 2
  kmeans = KMeans(n_clusters=k, random_state=1)
  # Fit the model to the data
  df_without_labels = X
  # Predict the cluster labels
  clusters = kmeans.fit_predict(df_without_labels)
  return clusters


# apply dbscan to the dataset
def dbscan_model(X):
  dbscan = DBSCAN(eps=0.5, min_samples=2)
  clusters = dbscan.fit_predict(X)
  return clusters


# using silhouette score -> measuring success of clustering results
# -1 TO 1 range
# -1: incorrect clustering, 0: overlapping clusters, 1: good clustering
def silhouette_score_analysis(X):
  df_without_labels = X
  silhouette_scores = []
  for k in range(2, 11):
      kmeans = KMeans(n_clusters=k, random_state=1)
      clusters = kmeans.fit_predict(df_without_labels)
      silhouette_scores.append(silhouette_score(df_without_labels, clusters))
      print(f"Silhouette Score for {k} clusters is {silhouette_score(df_without_labels, clusters)}")
  plt.plot(range(2, 11), silhouette_scores)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Silhouette Score')
  plt.show()