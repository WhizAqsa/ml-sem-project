import pandas as pd
from src.utils import fit_and_get_predictions,get_best_params_grid_search
from scripts.preprocess import preprocess_data
from xgboost import XGBClassifier
from src.evaluation import get_eval_metrics
from src.plots import visualize_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def run_xgb_model():
    X_train_scaled_2021_22,X_test_scaled_2021_22, y_train_encoded_2021_22,y_test_encoded_2021_22,X_train_scaled_2021_23, X_test_scaled_2021_23, y_test_encoded_2021_23,y_train_encoded_2021_23, X_test_scaled_2022_23, y_test_encoded_2022_23,X_train_scaled_2022_23,y_train_encoded_2022_23, X_scaled_21, y_encoded_21, X_scaled_22, y_encoded_22, X_scaled_23, y_encoded_23 = preprocess_data()   
    
    # save the preprocessed data
    print("Saving the preprocessed data...")

    # Combine and add labels for 2021-22
    combined_df_2021_22 = pd.concat([pd.DataFrame(X_train_scaled_2021_22), pd.DataFrame(X_test_scaled_2021_22)], ignore_index=True)
    combined_labels_2021_22 = pd.concat([pd.Series(y_train_encoded_2021_22), pd.Series(y_test_encoded_2021_22)], ignore_index=True)
    combined_df_2021_22['label'] = combined_labels_2021_22

    # Combine and add labels for 2021-23
    combined_df_2021_23 = pd.concat([pd.DataFrame(X_train_scaled_2021_23), pd.DataFrame(X_test_scaled_2021_23)], ignore_index=True)
    combined_labels_2021_23 = pd.concat([pd.Series(y_train_encoded_2021_23), pd.Series(y_test_encoded_2021_23)], ignore_index=True)
    combined_df_2021_23['label'] = combined_labels_2021_23

    # Combine and add labels for 2022-23
    combined_df_2022_23 = pd.concat([pd.DataFrame(X_train_scaled_2022_23), pd.DataFrame(X_test_scaled_2022_23)], ignore_index=True)
    combined_df_2021_22.to_csv('data/processed/cotton_rice_2021_22.csv', index=False)
    combined_df_2021_23.to_csv('data/processed/cotton_rice_2021_23.csv', index=False)
    combined_df_2022_23.to_csv('data/processed/cotton_rice_2022_23.csv', index=False)
    
    #---------------------1st Split---------------------#
    print("---------------------------------------------------")
    print("Train: 2021-22 and Test: 2023 (rice,cotton) dataset")
    print("-----------------------------------------------------")
    # run the default 
    print("Running default xgb model...")
    model = XGBClassifier(random_state=42)
    y_pred_xgb = fit_and_get_predictions(model, X_train_scaled_2021_22, y_train_encoded_2021_22, X_scaled_23)
    
    # print the classification report
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_xgb)
    print(f"Accuracy score for default xgb model: {score}")
    print(f"Classification report for default xgb model:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix...")
    visualize_confusion_matrix(cm)
    
    # grid search
    print("XGBoost Grid Search")
    params_grid_xgboost = {
    "max_depth": [3 ,5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "n_estimators": [100, 200, 300]
    }
    best_params_xgb = get_best_params_grid_search(
    XGBClassifier(random_state=42,verbosity=2),
    X_train_scaled_2021_22,
    y_train_encoded_2021_22,
    params_grid_xgboost
    )
    print(f"XGBoost Best params: {best_params_xgb}")
    best_xgb_model = XGBClassifier(
    random_state=42,
    **best_params_xgb
    )
    
    print("Running the best xgb model...")
    y_pred_xgb_best = fit_and_get_predictions(
    best_xgb_model,
    X_train_scaled_2021_22, 
    y_train_encoded_2021_22,
    X_scaled_23
    )
    
    # print the classification report for the best model
    score, eval_report, cm = get_eval_metrics(y_encoded_23, y_pred_xgb_best)
    print(f"Accuracy score for the best xgb model: {score}")
    print(f"Classification report for the best xgb model:\n{eval_report}")
    
    # visualise the conf matrix
    print("Visualise the confusion matrix after applying grid search...")
    visualize_confusion_matrix(cm)
    

if __name__ == "__main__":
    run_xgb_model()