import pandas as pd
from src.utils import fit_and_get_predictions, get_best_params_grid_search
from scripts.preprocess import preprocess_data
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from src.evaluation import get_eval_metrics
from src.plots import visualize_confusion_matrix


def run_bagging_model():
    
    X_train_scaled_2021_22,X_test_scaled_2021_22, y_train_encoded_2021_22,y_test_encoded_2021_22,X_train_scaled_2021_23, X_test_scaled_2021_23, y_test_encoded_2021_23,y_train_encoded_2021_23, X_test_scaled_2022_23, y_test_encoded_2022_23,X_train_scaled_2022_23,y_train_encoded_2022_23, X_scaled_21, y_encoded_21, X_scaled_22, y_encoded_22, X_scaled_23, y_encoded_23 = preprocess_data()   
    
    #-------------------------1st Split----------------------
    print("---------------------------------------------------")
    print("Train: 2021-22 and Test: 2023 (rice,cotton) dataset")
    print("--------------------------------------------------")
    # run the default 
    print("Running default bagging classifier...")
    model = BaggingClassifier(random_state=42)
    y_pred_rf = fit_and_get_predictions(model, X_train_scaled_2021_22, y_train_encoded_2021_22, X_scaled_23)
    
    # print the classification report
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_rf)
    print(f"Accuracy score for default bagging classifier: {score}")
    print(f"Classification report for default bagging classifier:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix...")
    visualize_confusion_matrix(cm)
    
    # grid search
    print("Bagging Classifier Grid Search")
    params_grid_bagging = {
    'n_estimators': [10, 20,30],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0, 2],
    }
    best_params_bagging = get_best_params_grid_search(
    BaggingClassifier(random_state=42),
    X_train_scaled_2021_22,
    y_train_encoded_2021_22,
    params_grid_bagging
    )
    print(f"rfoost Best params: {best_params_bagging}")
    best_bagging_model = BaggingClassifier(
    random_state=42,
    **best_params_bagging
    )
    
    print("Running the best bagging classifier...")
    y_pred_bagging_best = fit_and_get_predictions(
    best_bagging_model,
    X_train_scaled_2021_22, 
    y_train_encoded_2021_22,
    X_scaled_23
    )
    
    # print the classification report for the best model
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_bagging_best)
    print(f"Accuracy score for the best bagging classifier: {score}")
    print(f"Classification report for the best bagging classifier:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix after applying grid search...")
    visualize_confusion_matrix(cm)
    

if __name__ == "__main__":
    run_bagging_model()