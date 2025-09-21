from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from flask import jsonify

def handle_sklearn_operation(algorithm, df, params=None):
    """Handle all scikit-learn operations"""
    if params is None:
        params = {}

    try:
        target_col = params.get('target_column')

        if not target_col or target_col not in df.columns:
            return {'error': 'Target column not found or not specified'}

        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features for certain algorithms
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        result = {}

        if algorithm == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            result = {
                'type': 'regression',
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_),
                'predictions': y_pred[:10].tolist(),  # First 10 predictions
                'actual_values': y_test[:10].tolist()  # First 10 actual values
            }

        elif algorithm == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

            accuracy = accuracy_score(y_test, y_pred)

            result = {
                'type': 'classification',
                'accuracy': float(accuracy),
                'predictions': y_pred[:10].tolist(),
                'actual_values': y_test[:10].tolist(),
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_.tolist(),
                'classes': np.unique(y).tolist()
            }

            if y_pred_proba is not None:
                result['probabilities'] = y_pred_proba[:10].tolist()

        elif algorithm == 'decision_tree':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = DecisionTreeClassifier(random_state=42, max_depth=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                result = {
                    'type': 'classification',
                    'accuracy': float(accuracy),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
                    'max_depth': model.get_depth(),
                    'n_leaves': model.get_n_leaves(),
                    'classification_report': report
                }
            else:  # Regression
                model = DecisionTreeRegressor(random_state=42, max_depth=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                result = {
                    'type': 'regression',
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'r2_score': float(r2),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
                    'max_depth': model.get_depth(),
                    'n_leaves': model.get_n_leaves()
                }

        elif algorithm == 'random_forest':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                result = {
                    'type': 'classification',
                    'accuracy': float(accuracy),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth,
                    'classification_report': report
                }
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                result = {
                    'type': 'regression',
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'r2_score': float(r2),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist())),
                    'n_estimators': model.n_estimators,
                    'max_depth': model.max_depth
                }

        elif algorithm == 'svm':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = SVC(random_state=42, kernel='rbf', C=1.0)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                result = {
                    'type': 'classification',
                    'accuracy': float(accuracy),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'kernel': model.kernel,
                    'C': model.C,
                    'support_vectors': model.n_support_.tolist() if hasattr(model, 'n_support_') else None,
                    'classification_report': report
                }
            else:  # Regression
                model = SVR(kernel='rbf', C=1.0)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                result = {
                    'type': 'regression',
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'r2_score': float(r2),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'kernel': model.kernel,
                    'C': model.C
                }

        elif algorithm == 'knn':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                result = {
                    'type': 'classification',
                    'accuracy': float(accuracy),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'n_neighbors': model.n_neighbors,
                    'weights': model.weights,
                    'algorithm': model.algorithm,
                    'classification_report': report
                }
            else:  # Regression
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                result = {
                    'type': 'regression',
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'r2_score': float(r2),
                    'predictions': y_pred[:10].tolist(),
                    'actual_values': y_test[:10].tolist(),
                    'n_neighbors': model.n_neighbors,
                    'weights': model.weights,
                    'algorithm': model.algorithm
                }

        else:
            result = {'error': f"Algorithm '{algorithm}' not implemented"}

        return {'result': result}

    except Exception as e:
        return {'error': str(e)}