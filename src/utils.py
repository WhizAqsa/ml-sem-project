from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def separate_features_label(dataset):
    X = dataset.drop('label', axis=1)
    y = dataset['label']
    return X, y
        
    
# merege the two datasets
def merge_datasets(dataset1, dataset2):
    # adding a label column in both the datasets
    dataset1['label'] = 'cotton'
    dataset2['label'] = 'rice'
    # merge the two datasets
    merged_dataset = pd.concat([dataset1, dataset2], ignore_index=True)
    return merged_dataset
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
    
def plot_histogram(dataset):
  dataset.hist(bins=50, figsize=(20, 15))
  plt.show()
  

def plot_boxplot(dataset):
  figure = plt.figure(figsize=(12, 10))
  dataset.boxplot()
  plt.show()
  
  
def evaluate_model(y_test, y_pred):
  score = accuracy_score(y_test, y_pred)
  print(f"Accuracy Score: {score}")
  classification_report = classification_report(y_test, y_pred)
  print(f"Classification Report:\n{classification_report}")
  # confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  print(f"Confusion Matrix:\n{cm}")
  return score, classification_report,cm


def visualize_confusion_matrix(cm):
  plt.figure(figsize=(8, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cotton', 'Rice'], yticklabels=['Cotton', 'Rice'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()
  
  
def pca_analysis(X,y):
  kmeans = KMeans(n_clusters=2, random_state=1)
  pca = PCA(n_components=2)
  pca_features = pca.fit_transform(X)
  # scatter plot of pca features with clusters
  new_labels = kmeans.labels_
  plt.scatter(pca_features[:, 0], pca_features[:, 1], c=new_labels, cmap='viridis')
  plt.xlabel('PCA 1')
  plt.ylabel('PCA 2')
  plt.title('K-Means Clustering')
  plt.show()
  
  
# compare original dataset with predicted clusters
# Plot the identified clusters and compare
def compare_clusters(X,y):
  kmeans = KMeans(n_clusters=2, random_state=1)
  pca = PCA(n_components=2)
  pca_features = pca.fit_transform(X)
  new_labels = kmeans.labels_
  fig, axes = plt.subplots(1, 2, figsize=(12,7))
  axes[0].scatter(pca_features[:, 0], pca_features[:, 1], c=y, cmap='gist_rainbow', edgecolor='k', s=150)
  axes[1].scatter(pca_features[:, 0], pca_features[:, 1], c=new_labels, cmap='jet', edgecolor='k', s=150)
  axes[0].set_xlabel('PCA-1')
  axes[0].set_ylabel('PCA-2')
  axes[1].set_xlabel('PCA-1')
  axes[1].set_ylabel('PCA-2')
  axes[0].set_title('True Clusters')
  axes[1].set_title('Predicted Clusters')
  plt.show()
  
  
  
"""
Performance evaluation has been done via silhoutte score as it measures the similarity of each data point to its own clusters compared to other clusters and the plot visualises these scores for each sample.
"""

def cluster_purity(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(conf_matrix, axis=0)) / np.sum(conf_matrix)



def fit_and_get_predictions(model, X_train, y_train, X_test):
  model.fit(X_train, y_train)
  # make predictions
  y_pred = model.predict(X_test)
  return y_pred 