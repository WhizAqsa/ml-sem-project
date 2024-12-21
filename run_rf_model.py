import pandas as pd
from src.utils import fit_and_get_predictions, get_best_params_grid_search
from scripts.preprocess import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import get_eval_metrics
from src.plots import visualize_confusion_matrix, plot_feature_importance


def run_rf_model():
    # columns used to visualise feature importance
    columns = [
        "NDVI01",
        "NDVI02",
        "NDVI03",
        "NDVI04",
        "NDVI05",
        "NDVI06",
        "NDVI07",
        "NDVI08",
        "NDVI09",
        "NDVI10",
        "NDVI11",
        "NDVI12",
    ]
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
        print("Running default random forest model...")
        model = RandomForestClassifier(random_state=42)
        y_pred_rf = fit_and_get_predictions(model, X_train, y_train, X_test)

        # print the classification report
        score, eval_report, cm = get_eval_metrics(y_test, y_pred_rf)
        print(f"Accuracy score for default random forest model: {score}")
        print(f"Classification report for default random forest model:\n{eval_report}")

        # visualise the conf matrix
        print("Visualise the confusion matrix...")
        visualize_confusion_matrix(cm)

        # plot feature importance for the default model
        print("Plotting feature importance for the default model...")
        plot_feature_importance(model.feature_importances_, columns)

        # grid search
        print("Random Forest Grid Search")
        params_grid_rf = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4, 5, 7],
        }
        best_params_rf = get_best_params_grid_search(
            RandomForestClassifier(random_state=42),
            X_train,
            y_train,
            params_grid_rf,
        )
        print(f"Random Forest Best params: {best_params_rf}")
        best_rf_model = RandomForestClassifier(random_state=42, **best_params_rf)

        print("Running the best random forest model...")
        y_pred_rf_best = fit_and_get_predictions(
            best_rf_model, X_train, y_train, X_test
        )

        # print the classification report for the best model
        score, eval_report, cm = get_eval_metrics(y_test, y_pred_rf_best)
        print(f"Accuracy score for the best random forest model: {score}")
        print(f"Classification report for the best random forest model:\n{eval_report}")

        # visualise the conf matrix
        print("Visualise the confusion matrix after applying grid search...")
        visualize_confusion_matrix(cm)

        # plot feature importance for the best model
        print("Plotting feature importance for the best model...")
        plot_feature_importance(best_rf_model.feature_importances_, columns)


if __name__ == "__main__":
    run_rf_model()
