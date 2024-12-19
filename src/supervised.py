from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC



def fit_and_predict_xgboost(X_train,y_train,X_test):
  xgb_model = XGBClassifier(random_state=42)
  xgb_model.fit(X_train,y_train)
  # make predictions with XGBoost Classifier
  y_pred = xgb_model.predict(X_test)
  return y_pred

def random_forest_model(X_train,y_train,X_test):
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  # make predictions with Random Forest Classifier
  y_pred = rf_model.predict(X_test)
  return y_pred


def bagging_model(X_train,y_train,X_test):
  bagging_model = BaggingClassifier(n_estimators=100, random_state=42)
  bagging_model.fit(X_train, y_train)
  # make predictions with Bagging Classifier
  y_pred = bagging_model.predict(X_test)
  return y_pred


#


def fit_and_predict_svm(X_train,y_train,X_test):
  # Create an SVM classifier with the best hyperparameters
  model = SVC(kernel='rbf', C=10, gamma=1, random_state=42)
  model.fit(X_train, y_train)
  # make predictions with SVM Classifier
  y_pred = model.predict(X_test)
  return y_pred