from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import tensorflow as tf
from tensorflow import keras
import google.generativeai as genai
from openai import OpenAI
import os
from dotenv import load_dotenv
import tempfile
import uuid

# Load environment variables
load_dotenv()

# Configuration
USE_AI = os.getenv("USE_AI", "true").lower() == "true"

# Initialize OpenAI client only if AI is enabled
if USE_AI and os.getenv("OPEN_API_KEY"):
    openai_client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
else:
    openai_client = None
    print("AI features disabled. Set USE_AI=true and provide OPEN_API_KEY to enable.")

app = Flask(__name__)
CORS(app)

# Global data storage
data_store = {}

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Generate unique ID for this session
        session_id = str(uuid.uuid4())

        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        data_store[session_id] = df

        return jsonify({
            'session_id': session_id,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'preview': df.head(10).to_dict('records'),
            'stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pandas/<operation>', methods=['POST'])
def pandas_operations(operation):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        if operation == 'info':
            buffer = io.StringIO()
            df.info(buf=buffer)
            result = buffer.getvalue()
        elif operation == 'describe':
            result = df.describe().to_dict()
        elif operation == 'head':
            n = params.get('n', 5)
            result = df.head(n).to_dict('records')
        elif operation == 'tail':
            n = params.get('n', 5)
            result = df.tail(n).to_dict('records')
        elif operation == 'shape':
            result = df.shape
        elif operation == 'columns':
            result = list(df.columns)
        elif operation == 'dtypes':
            result = df.dtypes.astype(str).to_dict()
        elif operation == 'isnull':
            result = df.isnull().sum().to_dict()
        elif operation == 'corr':
            if df.select_dtypes(include=[np.number]).shape[1] > 1:
                result = df.corr().to_dict()
            else:
                result = "No numeric columns for correlation"
        elif operation == 'value_counts':
            col = params.get('column')
            if col and col in df.columns:
                result = df[col].value_counts().to_dict()
            else:
                result = "Column not found"
        elif operation == 'groupby':
            col = params.get('column')
            agg = params.get('agg', 'mean')
            if col and col in df.columns:
                result = df.groupby(col).agg(agg).to_dict('records') if len(df.columns) > 1 else "Need multiple columns"
            else:
                result = "Column not found"
        elif operation == 'pivot_table':
            values = params.get('values')
            index = params.get('index')
            if values and index and values in df.columns and index in df.columns:
                result = pd.pivot_table(df, values=values, index=index).to_dict()
            else:
                result = "Invalid parameters"
        elif operation == 'dropna':
            subset = params.get('subset')
            df_clean = df.dropna(subset=subset) if subset else df.dropna()
            data_store[session_id] = df_clean
            result = {'shape': df_clean.shape, 'message': 'NaN values dropped'}
        elif operation == 'fillna':
            value = params.get('value', 0)
            df_filled = df.fillna(value)
            data_store[session_id] = df_filled
            result = {'shape': df_filled.shape, 'message': 'NaN values filled'}
        elif operation == 'drop_duplicates':
            df_unique = df.drop_duplicates()
            data_store[session_id] = df_unique
            result = {'shape': df_unique.shape, 'message': 'Duplicates dropped'}
        elif operation == 'sort_values':
            by = params.get('by')
            ascending = params.get('ascending', True)
            if by and by in df.columns:
                df_sorted = df.sort_values(by=by, ascending=ascending)
                data_store[session_id] = df_sorted
                result = {'message': f'Data sorted by {by}'}
            else:
                result = "Column not found"
        elif operation == 'reset_index':
            df_reset = df.reset_index(drop=True)
            data_store[session_id] = df_reset
            result = {'message': 'Index reset'}
        elif operation == 'set_index':
            col = params.get('column')
            if col and col in df.columns:
                df_indexed = df.set_index(col)
                data_store[session_id] = df_indexed
                result = {'message': f'Index set to {col}'}
            else:
                result = "Column not found"
        elif operation == 'rename':
            columns = params.get('columns', {})
            df_renamed = df.rename(columns=columns)
            data_store[session_id] = df_renamed
            result = {'message': 'Columns renamed'}
        elif operation == 'merge':
            # This would require another dataset, simplified version
            result = "Merge operation requires additional dataset"
        else:
            result = f"Operation '{operation}' not implemented"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/numpy/<operation>', methods=['POST'])
def numpy_operations(operation):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return jsonify({'error': 'No numeric data available'}), 400

        arr = numeric_df.values

        if operation == 'mean':
            axis = params.get('axis', None)
            result = np.mean(arr, axis=axis).tolist()
        elif operation == 'median':
            axis = params.get('axis', None)
            result = np.median(arr, axis=axis).tolist()
        elif operation == 'std':
            axis = params.get('axis', None)
            result = np.std(arr, axis=axis).tolist()
        elif operation == 'var':
            axis = params.get('axis', None)
            result = np.var(arr, axis=axis).tolist()
        elif operation == 'min':
            axis = params.get('axis', None)
            result = np.min(arr, axis=axis).tolist()
        elif operation == 'max':
            axis = params.get('axis', None)
            result = np.max(arr, axis=axis).tolist()
        elif operation == 'sum':
            axis = params.get('axis', None)
            result = np.sum(arr, axis=axis).tolist()
        elif operation == 'prod':
            axis = params.get('axis', None)
            result = np.prod(arr, axis=axis).tolist()
        elif operation == 'cumsum':
            axis = params.get('axis', 0)
            result = np.cumsum(arr, axis=axis).tolist()
        elif operation == 'cumprod':
            axis = params.get('axis', 0)
            result = np.cumprod(arr, axis=axis).tolist()
        elif operation == 'sort':
            axis = params.get('axis', 0)
            result = np.sort(arr, axis=axis).tolist()
        elif operation == 'unique':
            result = np.unique(arr).tolist()
        elif operation == 'transpose':
            result = arr.T.tolist()
        elif operation == 'reshape':
            shape = params.get('shape')
            if shape:
                result = arr.reshape(shape).tolist()
            else:
                result = "Shape parameter required"
        elif operation == 'flatten':
            result = arr.flatten().tolist()
        elif operation == 'dot':
            # Simplified dot product with itself
            result = np.dot(arr.T, arr).tolist()
        elif operation == 'linalg_inv':
            if arr.shape[0] == arr.shape[1]:
                result = np.linalg.inv(arr).tolist()
            else:
                result = "Matrix must be square"
        elif operation == 'linalg_eig':
            if arr.shape[0] == arr.shape[1]:
                eigenvals, eigenvecs = np.linalg.eig(arr)
                result = {'eigenvalues': eigenvals.tolist(), 'eigenvectors': eigenvecs.tolist()}
            else:
                result = "Matrix must be square"
        elif operation == 'fft':
            result = np.fft.fft(arr[:, 0]).tolist() if arr.shape[1] > 0 else "No data"
        elif operation == 'sin':
            result = np.sin(arr).tolist()
        elif operation == 'cos':
            result = np.cos(arr).tolist()
        elif operation == 'exp':
            result = np.exp(arr).tolist()
        elif operation == 'log':
            result = np.log(np.abs(arr) + 1e-10).tolist()  # Avoid log(0)
        elif operation == 'sqrt':
            result = np.sqrt(np.abs(arr)).tolist()
        else:
            result = f"Operation '{operation}' not implemented"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize/<library>/<chart_type>', methods=['POST'])
def create_visualization(library, chart_type):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        if library == 'matplotlib':
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == 'histogram':
                col = params.get('column')
                if col and col in df.columns:
                    df[col].hist(ax=ax)
                    ax.set_title(f'Histogram of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
            elif chart_type == 'scatter':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    ax.scatter(df[x_col], df[y_col])
                    ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            elif chart_type == 'line':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    ax.plot(df[x_col], df[y_col])
                    ax.set_title(f'Line Plot: {x_col} vs {y_col}')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            elif chart_type == 'bar':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    ax.bar(df[x_col], df[y_col])
                    ax.set_title(f'Bar Chart: {x_col} vs {y_col}')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            elif chart_type == 'box':
                col = params.get('column')
                if col and col in df.columns:
                    ax.boxplot(df[col].dropna())
                    ax.set_title(f'Box Plot of {col}')
                    ax.set_ylabel(col)

            # Save plot to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return jsonify({'image': f'data:image/png;base64,{image_base64}'})

        elif library == 'seaborn':
            plt.figure(figsize=(10, 6))

            if chart_type == 'heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
                    plt.title('Correlation Heatmap')
            elif chart_type == 'pairplot':
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] > 1:
                    # Create a simpler version for web
                    fig, axes = plt.subplots(numeric_df.shape[1], numeric_df.shape[1], figsize=(12, 12))
                    for i in range(numeric_df.shape[1]):
                        for j in range(numeric_df.shape[1]):
                            if i == j:
                                axes[i, j].hist(numeric_df.iloc[:, i], bins=20)
                            else:
                                axes[i, j].scatter(numeric_df.iloc[:, j], numeric_df.iloc[:, i], alpha=0.5)
                    plt.tight_layout()
            elif chart_type == 'boxplot':
                col = params.get('column')
                if col and col in df.columns:
                    sns.boxplot(y=df[col])
                    plt.title(f'Box Plot of {col}')
            elif chart_type == 'violinplot':
                col = params.get('column')
                if col and col in df.columns:
                    sns.violinplot(y=df[col])
                    plt.title(f'Violin Plot of {col}')
            elif chart_type == 'histplot':
                col = params.get('column')
                if col and col in df.columns:
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Histogram of {col}')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

            return jsonify({'image': f'data:image/png;base64,{image_base64}'})

        elif library == 'plotly':
            if chart_type == 'scatter':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot: {x_col} vs {y_col}')
            elif chart_type == 'line':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.line(df, x=x_col, y=y_col, title=f'Line Plot: {x_col} vs {y_col}')
            elif chart_type == 'bar':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    fig = px.bar(df, x=x_col, y=y_col, title=f'Bar Chart: {x_col} vs {y_col}')
            elif chart_type == 'histogram':
                col = params.get('column')
                if col and col in df.columns:
                    fig = px.histogram(df, x=col, title=f'Histogram of {col}')
            elif chart_type == 'box':
                col = params.get('column')
                if col and col in df.columns:
                    fig = px.box(df, y=col, title=f'Box Plot of {col}')
            elif chart_type == 'heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    fig = px.imshow(numeric_df.corr(), title='Correlation Heatmap')
            elif chart_type == 'pie':
                col = params.get('column')
                if col and col in df.columns:
                    value_counts = df[col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index, title=f'Pie Chart of {col}')
            else:
                return jsonify({'error': f'Chart type {chart_type} not supported'}), 400

            return jsonify({'plotly_json': json.dumps(fig, cls=PlotlyJSONEncoder)})

        return jsonify({'error': f'Library {library} not supported'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sklearn/<algorithm>', methods=['POST'])
def sklearn_operations(algorithm):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        target_col = params.get('target_column')

        if not target_col or target_col not in df.columns:
            return jsonify({'error': 'Target column not found'}), 400

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if algorithm == 'linear_regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            result = {
                'mse': mse,
                'rmse': rmse,
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_,
                'predictions': y_pred[:10].tolist()  # First 10 predictions
            }
        elif algorithm == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            result = {
                'accuracy': accuracy,
                'predictions': y_pred[:10].tolist(),
                'coefficients': model.coef_.tolist(),
                'intercept': model.intercept_.tolist()
            }
        elif algorithm == 'decision_tree':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                result = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'predictions': y_pred[:10].tolist()
                }
            else:  # Regression
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    'type': 'regression',
                    'mse': mse,
                    'predictions': y_pred[:10].tolist()
                }
        elif algorithm == 'random_forest':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = RandomForestClassifier(random_state=42, n_estimators=100)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                result = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'predictions': y_pred[:10].tolist()
                }
            else:  # Regression
                model = RandomForestRegressor(random_state=42, n_estimators=100)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    'type': 'regression',
                    'mse': mse,
                    'predictions': y_pred[:10].tolist()
                }
        elif algorithm == 'svm':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = SVC(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                result = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'predictions': y_pred[:10].tolist()
                }
            else:  # Regression
                model = SVR()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    'type': 'regression',
                    'mse': mse,
                    'predictions': y_pred[:10].tolist()
                }
        elif algorithm == 'knn':
            if y.dtype == 'object' or len(y.unique()) < 10:  # Classification
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                result = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'predictions': y_pred[:10].tolist()
                }
            else:  # Regression
                model = KNeighborsRegressor(n_neighbors=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                result = {
                    'type': 'regression',
                    'mse': mse,
                    'predictions': y_pred[:10].tolist()
                }
        else:
            result = f"Algorithm '{algorithm}' not implemented"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tensorflow/<model_type>', methods=['POST'])
def tensorflow_operations(model_type):
    try:
        # Clear any previous TensorFlow sessions to prevent "out of session" errors
        tf.keras.backend.clear_session()

        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        target_col = params.get('target_column')

        if not target_col or target_col not in df.columns:
            return jsonify({'error': 'Target column not found'}), 400

        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Normalize data
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'simple_nn':
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif model_type == 'deep_nn':
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif model_type == 'cnn_1d':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model = keras.Sequential([
                keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        elif model_type == 'lstm':
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model = keras.Sequential([
                keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            return jsonify({'error': f'Model type {model_type} not supported'}), 400

        # Train model
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test).flatten()

        result = {
            'loss': loss,
            'mae': mae,
            'predictions': y_pred[:10].tolist(),
            'training_history': {
                'loss': history.history['loss'][-10:],  # Last 10 epochs
                'val_loss': history.history['val_loss'][-10:]
            }
        }

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai_summary', methods=['POST'])
def ai_summary():
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        # Prepare data summary
        summary = f"Dataset: {df.shape} | Columns: {list(df.columns)} | Types: {df.dtypes.to_dict()}"

        print(f"AI Summary request for session {session_id}")
        print(f"Dataset shape: {df.shape}")
        print(f"OpenAI API Key loaded: {'Yes' if os.getenv('OPEN_API_KEY') else 'No'}")

        # Check if AI is enabled
        if not USE_AI or not openai_client:
            fallback_summary = generate_basic_analysis(df)
            return jsonify({
                'summary': f'AI disabled. Basic analysis:\n\n{fallback_summary}\n\nEnable with USE_AI=true and OPEN_API_KEY'
            })

        # Use OpenAI to generate insights with fallback options
        try:
            prompt = f"Analyze this dataset summary and provide key insights, patterns, and recommendations:\n{summary}"
            print(f"Sending prompt to OpenAI: {prompt[:200]}...")

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Provide clear, actionable insights about datasets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            print(f"OpenAI response received: {response is not None}")

            if response and response.choices and len(response.choices) > 0:
                summary_text = response.choices[0].message.content
                print(f"Response text length: {len(summary_text)}")
                return jsonify({'summary': summary_text})
            else:
                print("Empty response from OpenAI")
                return jsonify({'summary': 'AI summary generation returned empty response. Please try again.'})
        except Exception as ai_error:
            print(f"AI Error: {str(ai_error)}")

            error_msg = str(ai_error).lower()
            if 'api' in error_msg and 'key' in error_msg:
                return jsonify({'summary': 'OpenAI API key error. Check .env file.'})
            elif 'quota' in error_msg or 'limit' in error_msg:
                fallback_summary = generate_basic_analysis(df)
                return jsonify({
                    'summary': f'API quota exceeded. Basic analysis:\n\n{fallback_summary}\n\nAdd credits or use different key.'
                })
            elif 'model' in error_msg:
                return jsonify({'summary': 'Model not available. Check OpenAI access.'})
            else:
                return jsonify({'summary': f'AI failed: {str(ai_error)}. Check API key.'})

    except Exception as e:
        print(f"General Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<session_id>', methods=['GET'])
def download_file(session_id):
    try:
        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name

        return send_file(temp_file, as_attachment=True, download_name='processed_data.csv')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_ai', methods=['GET'])
def test_ai():
    try:
        # Test if OpenAI API key is loaded
        api_key = os.getenv('OPEN_API_KEY')
        if not api_key:
            return jsonify({'status': 'error', 'message': 'OpenAI API key not found in environment'})

        # Test OpenAI connection
        try:
            print("Testing OpenAI connection...")

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a test message."}],
                max_tokens=50
            )

            if response and response.choices and len(response.choices) > 0:
                return jsonify({
                    'status': 'success',
                    'message': 'OpenAI connection working',
                    'model_used': 'gpt-3.5-turbo',
                    'response_length': len(response.choices[0].message.content)
                })
            else:
                return jsonify({'status': 'error', 'message': 'Empty response from OpenAI'})
        except Exception as ai_error:
            return jsonify({'status': 'error', 'message': f'OpenAI connection failed: {str(ai_error)}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def generate_basic_analysis(df):
    """Generate basic statistical analysis when AI is unavailable"""
    try:
        analysis = []
        analysis.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        analysis.append(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        dtypes_count = df.dtypes.value_counts()
        analysis.append(f"Data Types: {dict(dtypes_count)}")

        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            analysis.append(f"Missing values: {missing_total} total")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis.append(f"Numeric columns: {len(numeric_cols)}")
            for col in numeric_cols[:3]:
                stats = df[col].describe()
                analysis.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            analysis.append(f"Categorical columns: {len(cat_cols)}")

        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr()
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if not np.isnan(corr_val):
                            corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_val))
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: x[2], reverse=True)
                    analysis.append("Top correlations:")
                    for col1, col2, corr in corr_pairs[:2]:
                        analysis.append(f"  {col1} ↔ {col2}: {corr:.3f}")
            except:
                pass

        analysis.append(f"Size: {'Small' if df.shape[0] < 1000 else 'Medium' if df.shape[0] < 10000 else 'Large'}")
        return "\n".join(analysis)

    except Exception as e:
        return f"Analysis failed: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)