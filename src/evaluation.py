from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 

def evaluate_model(y_test, y_pred):
  score = accuracy_score(y_test, y_pred)
  # confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  # classification report
  eval_report = classification_report(y_test, y_pred)
  return score, eval_report,cm