from matplotlib import pyplot as plt
import seaborn as sns

def plot_histogram(dataset):
  dataset.hist(bins=50, figsize=(20, 15))
  plt.show()
  

def plot_boxplot(dataset):
  figure = plt.figure(figsize=(12, 10))
  dataset.boxplot()
  plt.show()
  
  
def visualize_confusion_matrix(cm):
  plt.figure(figsize=(8, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cotton', 'Rice'], yticklabels=['Cotton', 'Rice'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()