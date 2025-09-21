from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import tempfile
import uuid
import os
import requests
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv

# Import our modular functions
from pandas_operations import handle_pandas_operation
from numpy_operations import handle_numpy_operation
from visualization import create_visualization
from sklearn_operations import handle_sklearn_operation
from tensorflow_operations import handle_tensorflow_operation
from ai_summary import generate_ai_summary
from report_generator import report_generator_service
from dashboard import dashboard_service

# Load environment variables
load_dotenv()

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
        result = handle_pandas_operation(operation, df, params)

        return jsonify(result)

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
        result = handle_numpy_operation(operation, df, params)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize/<library>/<chart_type>', methods=['POST'])
def create_visualization_endpoint(library, chart_type):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        result = create_visualization(library, chart_type, df, params)

        return jsonify(result)

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
        result = handle_sklearn_operation(algorithm, df, params)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tensorflow/<model_type>', methods=['POST'])
def tensorflow_operations(model_type):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        params = data.get('params', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        result = handle_tensorflow_operation(model_type, df, params)

        return jsonify(result)

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
        result = generate_ai_summary(df, session_id)

        return jsonify(result)

    except Exception as e:
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

@app.route('/reports/create', methods=['POST'])
def create_report():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        report_type = data.get('report_type', 'html')
        title = data.get('title', 'Data Analysis Report')
        include_stats = data.get('include_stats', True)
        include_correlations = data.get('include_correlations', True)
        include_visualizations = data.get('include_visualizations', True)
        include_ai = data.get('include_ai', True)

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        # Get AI insights if requested
        ai_insights = ""
        if include_ai:
            try:
                ai_result = generate_ai_summary(df, session_id)
                ai_insights = ai_result.get('summary', '')
            except:
                ai_insights = "AI insights not available"

        result = report_generator_service.create_report(
            df, report_type, title, include_stats, include_correlations,
            include_visualizations, include_ai, ai_insights
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reports/download/<report_id>', methods=['GET'])
def download_report(report_id):
    try:
        return report_generator_service.get_report(report_id)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboards/create', methods=['POST'])
def create_dashboard():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        title = data.get('title', 'Data Dashboard')
        widgets = data.get('widgets', [])
        layout = data.get('layout', 'grid')

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]
        result = dashboard_service.create_dashboard(df, session_id, title, widgets, layout)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboards/<dashboard_id>', methods=['GET'])
def get_dashboard(dashboard_id):
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400

        result = dashboard_service.get_dashboard(dashboard_id, session_id)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboards/<dashboard_id>', methods=['PUT'])
def update_dashboard(dashboard_id):
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        title = data.get('title')
        widgets = data.get('widgets')
        layout = data.get('layout')

        result = dashboard_service.update_dashboard(dashboard_id, session_id, title, widgets, layout)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboards/<dashboard_id>', methods=['DELETE'])
def delete_dashboard(dashboard_id):
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400

        result = dashboard_service.delete_dashboard(dashboard_id, session_id)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboards', methods=['GET'])
def list_dashboards():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400

        result = dashboard_service.list_dashboards(session_id)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/widget/<widget_type>', methods=['POST'])
def render_dashboard_widget(widget_type):
    """Render a specific widget for dashboard display"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        widget_config = data.get('widget_config', {})

        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        # Auto-configure parameters if not provided
        if not widget_config or len(widget_config) == 0:
            widget_config = get_auto_widget_config(df, widget_type)

        # Generate visualization based on widget type
        if widget_type in ['histogram', 'scatter', 'line', 'bar', 'box', 'heatmap', 'pie']:
            # Map widget types to visualization library calls
            library_map = {
                'histogram': ('matplotlib', 'histogram'),
                'scatter': ('plotly', 'scatter'),
                'line': ('plotly', 'line'),
                'bar': ('plotly', 'bar'),
                'box': ('plotly', 'box'),
                'heatmap': ('seaborn', 'heatmap'),
                'pie': ('plotly', 'pie')
            }

            library, chart_type = library_map[widget_type]
            result = create_visualization(library, chart_type, df, widget_config)

            if result.get('image'):
                return jsonify({'type': 'image', 'data': result['image']})
            elif result.get('plotly_json'):
                return jsonify({'type': 'plotly', 'data': result['plotly_json']})
            else:
                return jsonify({'error': result.get('error', 'Failed to generate visualization')})

        return jsonify({'error': f'Unsupported widget type: {widget_type}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_auto_widget_config(df, widget_type):
    """Automatically configure widget parameters based on data"""
    config = {}

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if widget_type == 'histogram' and numeric_cols:
        config['column'] = numeric_cols[0]  # Use first numeric column

    elif widget_type in ['scatter', 'line', 'bar'] and len(numeric_cols) >= 2:
        config['x_column'] = numeric_cols[0]
        config['y_column'] = numeric_cols[1]

    elif widget_type == 'box' and numeric_cols:
        config['column'] = numeric_cols[0]

    elif widget_type == 'pie' and categorical_cols:
        config['column'] = categorical_cols[0]

    # heatmap and other types don't need specific column config

    return config

@app.route('/scrape', methods=['POST'])
def scrape_data():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        site = data.get('site', '').strip()
        num_pages = int(data.get('num_pages', 1))
        custom_filters = data.get('filters', [])

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        if num_pages < 1 or num_pages > 10:
            return jsonify({'error': 'Number of pages must be between 1 and 10'}), 400

        # Default filters if none provided
        if not custom_filters:
            custom_filters = [
                "database", "dataset", "repository", "data center", "platform",
                "aNANt", "MXene-DB", "Mem-ces", "nanoHUB", "materials database",
                "data-driven", "2D materials database", "AFLOW", "JARVIS",
                "C2DB", "Materials Project", "high-throughput", "first-principles",
                "data compilation", "benchmark", "machine learning"
            ]

        # Build search query
        if site:
            search_query = f"{query} site:{site}"
        else:
            search_query = query

        base_url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        results = []
        total_results = 0

        for page in range(num_pages):
            params = {
                'q': search_query,
                's': page * 50
            }

            try:
                response = requests.post(base_url, data=params, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.select("a.result__a")

                for a_tag in links:
                    title = a_tag.get_text()
                    link = a_tag['href']
                    total_results += 1

                    # Check if title matches any filter
                    if any(f.lower() in title.lower() for f in custom_filters):
                        results.append({
                            'title': title.strip(),
                            'link': link.strip(),
                            'page': page + 1
                        })

                time.sleep(1.5)  # Rate limiting

            except Exception as e:
                return jsonify({'error': f'Error scraping page {page + 1}: {str(e)}'}), 500

        # Remove duplicates
        unique_results = []
        seen_links = set()
        for result in results:
            if result['link'] not in seen_links:
                unique_results.append(result)
                seen_links.add(result['link'])

        return jsonify({
            'total_scraped': total_results,
            'filtered_results': len(unique_results),
            'results': unique_results,
            'query': query,
            'site': site,
            'pages': num_pages,
            'filters_used': custom_filters
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tensorflow/download_code/<session_id>', methods=['GET'])
def download_tensorflow_code(session_id):
    """Download TensorFlow model code for the trained model"""
    try:
        if session_id not in data_store:
            return jsonify({'error': 'Session not found'}), 404

        df = data_store[session_id]

        # Get model parameters from query string
        model_type = request.args.get('model_type', 'simple_nn')
        target_column = request.args.get('target_column', 'target')
        epochs = int(request.args.get('epochs', 50))
        batch_size = int(request.args.get('batch_size', 32))

        # Generate Python code
        code = generate_tensorflow_code(model_type, target_column, epochs, batch_size, df)

        # Return code as downloadable file
        return send_file(
            io.BytesIO(code.encode()),
            as_attachment=True,
            download_name=f'tensorflow_{model_type}_model.py',
            mimetype='text/plain'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_tensorflow_code(model_type, target_column, epochs, batch_size, df):
    """Generate TensorFlow model code based on parameters"""

    # Get numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    feature_cols = numeric_cols[:5]  # Use first 5 numeric columns as features

    # Generate model layers based on type
    if model_type == 'simple_nn':
        model_layers = """    model.add(layers.Dense(64, activation='relu', input_shape=(len(feature_columns),)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification"""
    elif model_type == 'deep_nn':
        model_layers = """    model.add(layers.Dense(128, activation='relu', input_shape=(len(feature_columns),)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification"""
    elif model_type == 'cnn_1d':
        model_layers = """    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(len(feature_columns), 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification"""
    elif model_type == 'lstm':
        model_layers = """    model.add(layers.LSTM(64, input_shape=(len(feature_columns), 1), return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification"""
    else:
        model_layers = """    model.add(layers.Dense(64, activation='relu', input_shape=(len(feature_columns),)))
    model.add(layers.Dense(1, activation='sigmoid'))"""

    # Create a simple template-based code generation
    template = '''# Generated TensorFlow Model Code
# Model Type: MODEL_TYPE
# Generated by Data Hub
# Target Column: TARGET_COL
# Feature Columns: FEATURE_COLS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data():
    """Load and preprocess your dataset"""
    # Replace with your actual data loading
    # df = pd.read_csv('your_data.csv')

    # For demonstration, create sample data structure
    print("Loading data...")
    print("Target column: TARGET_COL")
    print("Feature columns: FEATURE_COLS")

    # Sample data structure (replace with your actual data)
    # X = df[feature_columns]
    # y = df['TARGET_COL']
    # return X, y
    pass

# Create MODEL_TYPE_UPPER model
def create_model(input_shape):
    """Create the neural network model"""
    model = keras.Sequential()

MODEL_LAYERS

    return model

# Training configuration
def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model with the provided data"""
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',  # Change based on your problem type
        metrics=['accuracy', 'precision', 'recall']
    )

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    return history

# Evaluation and visualization
def evaluate_model(model, X_test, y_test, history):
    """Evaluate model performance and create visualizations"""
    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Classification report
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\\nConfusion Matrix:")
    print(cm)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist()
    }

# Main execution
if __name__ == "__main__":
    print("TensorFlow MODEL_TITLE Model Training")
    print("=" * 50)

    # Load data
    X, y = load_data()

    if X is None or y is None:
        print("Please implement the load_data() function with your actual data")
        exit()

    # Preprocess data
    print("\\nPreprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training samples: {X_train.shape[0]}")
    print("Testing samples: {X_test.shape[0]}")
    print("Number of features: {X_train.shape[1]}")

    # Reshape for specific model types
    if 'MODEL_TYPE' in ['cnn_1d', 'lstm']:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        print("Reshaped for MODEL_TYPE: {X_train.shape}")

    # Create and train model
    print("\\nCreating model...")
    model = create_model((X_train.shape[1], 1))

    print("\\nTraining model...")
    print("Epochs: EPOCHS")
    print("Batch size: BATCH_SIZE")
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate model
    print("\\nEvaluating model...")
    evaluation_results = evaluate_model(model, X_test, y_test, history)

    # Save model
    model.save('trained_MODEL_TYPE_model.h5')
    print("\\nModel saved as 'trained_MODEL_TYPE_model.h5'")

    # Save scaler for future predictions
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")

    print("\\nTraining completed successfully!")
    print("Files saved:")
    print("- trained_MODEL_TYPE_model.h5 (model weights)")
    print("- scaler.pkl (data scaler)")
    print("- training_history.png (training plots)")
'''

    # Replace placeholders with actual values
    code = template.replace('MODEL_TYPE', model_type)
    code = code.replace('MODEL_TYPE_UPPER', model_type.replace('_', ' ').upper())
    code = code.replace('MODEL_TITLE', model_type.replace('_', ' ').title())
    code = code.replace('TARGET_COL', target_column)
    code = code.replace('FEATURE_COLS', ', '.join(feature_cols))
    code = code.replace('EPOCHS', str(epochs))
    code = code.replace('BATCH_SIZE', str(batch_size))
    code = code.replace('MODEL_LAYERS', model_layers)

    return code

@app.route('/test_ai', methods=['GET'])
def test_ai():
    try:
        # Test local AI analysis capabilities
        import pandas as pd
        import numpy as np

        # Create a small test dataset
        test_data = {
            'numeric_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'category_col': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'another_numeric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        test_df = pd.DataFrame(test_data)

        # Test if our analysis functions work
        try:
            from ai_summary import generate_basic_analysis
            result = generate_basic_analysis(test_df)

            if result and 'summary' in result and len(result['summary']) > 100:
                return jsonify({
                    'status': 'success',
                    'message': 'Local AI analysis ready',
                    'analysis_type': 'Advanced statistical analysis',
                    'features': 'Correlation analysis, outlier detection, data quality assessment'
                })
            else:
                return jsonify({'status': 'error', 'message': 'Analysis function returned insufficient results'})
        except Exception as analysis_error:
            return jsonify({'status': 'error', 'message': f'Analysis function failed: {str(analysis_error)}'})
            # Test Gemini
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key or gemini_key == "your_gemini_api_key_here":
                return jsonify({'status': 'error', 'message': 'Gemini API key not configured. Please set GEMINI_API_KEY in .env file.'})

            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Hello, this is a test message.")
                return jsonify({
                    'status': 'success',
                    'message': 'Gemini connection working',
                    'model_used': 'gemini-1.5-flash',
                    'response_length': len(response.text)
                })
            except Exception as ai_error:
                return jsonify({'status': 'error', 'message': f'Gemini connection failed: {str(ai_error)}'})
        else:
            # Test OpenAI
            api_key = os.getenv('OPEN_API_KEY')
            if not api_key or api_key == "your_openai_api_key_here":
                return jsonify({'status': 'error', 'message': 'OpenAI API key not configured. Please set OPEN_API_KEY in .env file.'})

            try:
                from openai import OpenAI
                test_client = OpenAI(api_key=api_key)

                response = test_client.chat.completions.create(
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)