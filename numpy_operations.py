import numpy as np
from flask import jsonify

def handle_numpy_operation(operation, df, params=None):
    """Handle all numpy operations"""
    if params is None:
        params = {}

    try:
        # Get numeric columns from dataframe
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return {'error': 'No numeric data available for NumPy operations'}

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
                try:
                    result = arr.reshape(shape).tolist()
                except ValueError as e:
                    result = f"Cannot reshape array: {str(e)}"
            else:
                result = "Shape parameter required"
        elif operation == 'flatten':
            result = arr.flatten().tolist()
        elif operation == 'dot':
            # Simplified dot product with itself
            try:
                result = np.dot(arr.T, arr).tolist()
            except ValueError as e:
                result = f"Cannot compute dot product: {str(e)}"
        elif operation == 'linalg_inv':
            if arr.shape[0] == arr.shape[1]:
                try:
                    result = np.linalg.inv(arr).tolist()
                except np.linalg.LinAlgError as e:
                    result = f"Matrix inversion failed: {str(e)}"
            else:
                result = "Matrix must be square for inversion"
        elif operation == 'linalg_eig':
            if arr.shape[0] == arr.shape[1]:
                try:
                    eigenvals, eigenvecs = np.linalg.eig(arr)
                    result = {'eigenvalues': eigenvals.tolist(), 'eigenvectors': eigenvecs.tolist()}
                except np.linalg.LinAlgError as e:
                    result = f"Eigenvalue computation failed: {str(e)}"
            else:
                result = "Matrix must be square for eigenvalue computation"
        elif operation == 'fft':
            try:
                result = np.fft.fft(arr[:, 0]).tolist() if arr.shape[1] > 0 else "No data"
            except Exception as e:
                result = f"FFT computation failed: {str(e)}"
        elif operation == 'sin':
            result = np.sin(arr).tolist()
        elif operation == 'cos':
            result = np.cos(arr).tolist()
        elif operation == 'exp':
            result = np.exp(arr).tolist()
        elif operation == 'log':
            # Avoid log(0) by adding small epsilon
            result = np.log(np.abs(arr) + 1e-10).tolist()
        elif operation == 'sqrt':
            result = np.sqrt(np.abs(arr)).tolist()
        else:
            result = f"Operation '{operation}' not implemented"

        return {'result': result}

    except Exception as e:
        return {'error': str(e)}