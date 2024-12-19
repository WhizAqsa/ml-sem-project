from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import numpy as np
from sklearn import metrics



def get_eval_metrics(y_test, y_pred):
  score = accuracy_score(y_test, y_pred)
  # confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  # classification report
  eval_report = classification_report(y_test, y_pred)
  return score, eval_report,cm


def calculate_cluster_purity(y_true, y_pred):
  # Ensure y_true and y_pred are numpy arrays
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  
  # Find unique clusters
  unique_clusters = np.unique(y_pred)
  
  purity = 0
    
  # Loop through each cluster
  for cluster in unique_clusters:
    # Find indices of the current cluster
    cluster_indices = np.where(y_pred == cluster)[0]
    
    # Find the most frequent true label in the current cluster
    most_frequent_label = np.argmax(np.bincount(y_true[cluster_indices]))
    
    # Calculate the number of samples in the cluster that match the most frequent true label
    cluster_purity = np.sum(y_true[cluster_indices] == most_frequent_label)
    
    # Add the purity for this cluster to the total purity
    purity += cluster_purity / len(cluster_indices)
  
  # Calculate the average purity over all clusters
  purity /= len(unique_clusters)
  
  return purity

def get_confusion_matrix(y_true, y_pred):
  # find the unique labels
  unique_labels = np.unique(y_true)
  # initialize the confusion matrix
  cm = np.zeros((len(unique_labels), len(unique_labels)))
  # loop through each label
  for i, label in enumerate(unique_labels):
    # find the true labels that match the current label
    true_labels = y_true[y_true == label]
    # loop through each unique label
    for j, unique_label in enumerate(unique_labels):
      # find the predicted labels that match the current unique label
      predicted_labels = y_pred[y_true == unique_label]
      # find the number of matches between the true and predicted labels
      cm[i, j] = np.sum(true_labels == predicted_labels)
  return cm