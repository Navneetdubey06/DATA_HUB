try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from flask import jsonify

def handle_tensorflow_operation(model_type, df, params=None):
    if not TENSORFLOW_AVAILABLE:
        return {'error': 'TensorFlow is not available. Deep learning features require TensorFlow installation.'}
    """Handle all TensorFlow operations"""
    if params is None:
        params = {}

    # Clear any previous TensorFlow sessions to prevent "out of session" errors
    tf.keras.backend.clear_session()

    try:
        target_col = params.get('target_column')

        if not target_col or target_col not in df.columns:
            return {'error': 'Target column not found or not specified'}

        # Prepare data
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values

        # Normalize data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get training parameters
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)

        result = {}

        if model_type == 'simple_nn':
            # Simple Neural Network
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2,
                              verbose=0)

            # Evaluate
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test).flatten()

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                'model_type': 'Simple Neural Network',
                'architecture': [
                    f'Dense(64, relu) - Input: {X_train.shape[1]}',
                    'Dropout(0.2)',
                    'Dense(32, relu)',
                    'Dropout(0.2)',
                    'Dense(1, linear)'
                ],
                'loss': float(loss),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'r2_score': float(r2),
                'predictions': y_pred[:10].tolist(),
                'actual_values': y_test[:10].tolist(),
                'training_history': {
                    'epochs': list(range(1, len(history.history['loss']) + 1)),
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']]
                },
                'final_metrics': {
                    'train_loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'train_mae': float(history.history['mae'][-1]),
                    'val_mae': float(history.history['val_mae'][-1])
                }
            }

        elif model_type == 'deep_nn':
            # Deep Neural Network
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2,
                              verbose=0)

            # Evaluate
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test).flatten()

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                'model_type': 'Deep Neural Network',
                'architecture': [
                    f'Dense(128, relu) - Input: {X_train.shape[1]}',
                    'Dropout(0.3)',
                    'Dense(64, relu)',
                    'Dropout(0.2)',
                    'Dense(32, relu)',
                    'Dropout(0.2)',
                    'Dense(16, relu)',
                    'Dense(1, linear)'
                ],
                'loss': float(loss),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'r2_score': float(r2),
                'predictions': y_pred[:10].tolist(),
                'actual_values': y_test[:10].tolist(),
                'training_history': {
                    'epochs': list(range(1, len(history.history['loss']) + 1)),
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']]
                },
                'final_metrics': {
                    'train_loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'train_mae': float(history.history['mae'][-1]),
                    'val_mae': float(history.history['val_mae'][-1])
                }
            }

        elif model_type == 'cnn_1d':
            # 1D Convolutional Neural Network
            X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = keras.Sequential([
                keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(64, 3, activation='relu'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(X_train_cnn, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2,
                              verbose=0)

            # Evaluate
            loss, mae = model.evaluate(X_test_cnn, y_test, verbose=0)
            y_pred = model.predict(X_test_cnn).flatten()

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                'model_type': '1D Convolutional Neural Network',
                'architecture': [
                    f'Conv1D(32, 3, relu) - Input: ({X_train.shape[1]}, 1)',
                    'MaxPooling1D(2)',
                    'Conv1D(64, 3, relu)',
                    'MaxPooling1D(2)',
                    'Flatten',
                    'Dense(64, relu)',
                    'Dropout(0.2)',
                    'Dense(1, linear)'
                ],
                'loss': float(loss),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'r2_score': float(r2),
                'predictions': y_pred[:10].tolist(),
                'actual_values': y_test[:10].tolist(),
                'training_history': {
                    'epochs': list(range(1, len(history.history['loss']) + 1)),
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']]
                },
                'final_metrics': {
                    'train_loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'train_mae': float(history.history['mae'][-1]),
                    'val_mae': float(history.history['val_mae'][-1])
                }
            }

        elif model_type == 'lstm':
            # LSTM Network
            X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = keras.Sequential([
                keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(25, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train model
            history = model.fit(X_train_lstm, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=0.2,
                              verbose=0)

            # Evaluate
            loss, mae = model.evaluate(X_test_lstm, y_test, verbose=0)
            y_pred = model.predict(X_test_lstm).flatten()

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                'model_type': 'LSTM Neural Network',
                'architecture': [
                    f'LSTM(50, relu) - Input: ({X_train.shape[1]}, 1)',
                    'Dropout(0.2)',
                    'LSTM(25, relu)',
                    'Dropout(0.2)',
                    'Dense(16, relu)',
                    'Dense(1, linear)'
                ],
                'loss': float(loss),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'r2_score': float(r2),
                'predictions': y_pred[:10].tolist(),
                'actual_values': y_test[:10].tolist(),
                'training_history': {
                    'epochs': list(range(1, len(history.history['loss']) + 1)),
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']],
                    'mae': [float(x) for x in history.history['mae']],
                    'val_mae': [float(x) for x in history.history['val_mae']]
                },
                'final_metrics': {
                    'train_loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'train_mae': float(history.history['mae'][-1]),
                    'val_mae': float(history.history['val_mae'][-1])
                }
            }

        else:
            result = {'error': f"Model type '{model_type}' not implemented"}

        return {'result': result}

    except Exception as e:
        return {'error': str(e)}