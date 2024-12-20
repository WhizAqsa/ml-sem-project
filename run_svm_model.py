import pandas as pd
from src.utils import fit_and_get_predictions, get_best_params_grid_search
from scripts.preprocess import preprocess_data
from sklearn.svm import SVC
from src.evaluation import get_eval_metrics
from src.plots import visualize_confusion_matrix


def run_svm_model():
    (
        X_train_scaled_2021_22,
        X_test_scaled_2021_22,
        y_train_encoded_2021_22,
        y_test_encoded_2021_22,
        X_train_scaled_2021_23,
        X_test_scaled_2021_23,
        y_test_encoded_2021_23,
        y_train_encoded_2021_23,
        X_test_scaled_2022_23,
        y_test_encoded_2022_23,
        X_train_scaled_2022_23,
        y_train_encoded_2022_23,
        X_scaled_21,
        y_encoded_21,
        X_scaled_22,
        y_encoded_22,
        X_scaled_23,
        y_encoded_23,
    ) = preprocess_data()

    splits = [
        (
            "2021-22",
            X_train_scaled_2021_22,
            y_train_encoded_2021_22,
            X_scaled_23,
            y_encoded_23,
        ),
        (
            "2022-23",
            X_train_scaled_2022_23,
            y_train_encoded_2022_23,
            X_scaled_21,
            y_encoded_21,
        ),
        (
            "2021-23",
            X_train_scaled_2021_23,
            y_train_encoded_2021_23,
            X_scaled_22,
            y_encoded_22,
        ),
    ]

    for split_name, X_train, y_train, X_test, y_test in splits:
        test_year = (
            "2023"
            if split_name == "2021-22"
            else "2021"
            if split_name == "2022-23"
            else "2022"
        )
        print(f"---------------------------------------------------")
        print(f"Train: {split_name} and Test: {test_year} (rice, cotton) dataset")
        print(f"-----------------------------------------------------")

        # run the default
        print("Running default SVM model...")
        model = SVC(random_state=42)
        y_pred_rf = fit_and_get_predictions(model, X_train, y_train, X_test)

        # print the classification report
        score, eval_report, cm = get_eval_metrics(y_test, y_pred_rf)
        print(f"Accuracy score for default SVM model: {score}")
        print(f"Classification report for default SVM model:\n{eval_report}")

        # visualise the conf matrix
        print("Visualise the confusion matrix...")
        visualize_confusion_matrix(cm)

        # grid search
        # print("SVM Grid Search")
        # params_grid_svm = {
        #     "C": [0.1, 1, 3, 0.5],
        #     "gamma": [3, 0.1, 0.5, 1],
        #     "kernel": ["rbf", "linear", "sigmoid"],
        # }
        params_grid_svm = {
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1, 10],
            "kernel": ["rbf", "linear", "sigmoid"],
        }
        best_params_svm = get_best_params_grid_search(
            SVC(random_state=42), X_train, y_train, params_grid_svm
        )
        print(f"SVM Best params: {best_params_svm}")
        best_svm_model = SVC(random_state=42, **best_params_svm)

        print("Running the best SVM model...")
        y_pred_svm_best = fit_and_get_predictions(
            best_svm_model, X_train, y_train, X_test
        )

        # print the classification report for the best model
        score, eval_report, cm = get_eval_metrics(y_test, y_pred_svm_best)
        print(f"Accuracy score for the best SVM model: {score}")
        print(f"Classification report for the best SVM model:\n{eval_report}")

        # visualise the conf matrix
        print("Visualise the confusion matrix after applying grid search...")
        visualize_confusion_matrix(cm)


if __name__ == "__main__":
    run_svm_model()
