
from sklearn.model_selection import GridSearchCV

def separate_features_label(dataset):
    X = dataset.drop('label', axis=1)
    y = dataset['label']
    return X, y
  
  
def fit_and_get_predictions(model, X_train, y_train, X_test):
  model.fit(X_train, y_train)
  # make predictions
  y_pred = model.predict(X_test)
  return y_pred 


# Perform grid search with cross-validation
def get_best_params_grid_search(model, X_train, y_train, param_grid):
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro', verbose=2)
  grid_search.fit(X_train, y_train)
  return grid_search.best_params_