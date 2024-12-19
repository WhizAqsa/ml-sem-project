from src.unsupervised import kmeans_model, dbscan_model, elbow_method, silhouette_score_analysis

def run_unsupervised_models(X):
    # Elbow Method
    elbow_method(X)
    # KMeans Clustering
    clusters_kmeans = kmeans_model(X)
    # DBSCAN Clustering
    clusters_dbscan = dbscan_model(X)
    # Silhouette Score Analysis
    return clusters_kmeans, clusters_dbscan, silhouette_score_analysis(X)
