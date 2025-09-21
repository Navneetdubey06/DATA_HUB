import pandas as pd
import numpy as np
from flask import jsonify

def handle_pandas_operation(operation, df, params=None):
    """Handle all pandas operations"""
    if params is None:
        params = {}

    try:
        if operation == 'info':
            import io
            from contextlib import redirect_stdout
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                df.info()
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
                result = getattr(df.groupby(col), agg)().to_dict('records') if len(df.columns) > 1 else "Need multiple columns"
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
            # Note: This would modify the original dataframe in a real implementation
            result = {'shape': df_clean.shape, 'message': 'NaN values dropped'}
        elif operation == 'fillna':
            value = params.get('value', 0)
            df_filled = df.fillna(value)
            # Note: This would modify the original dataframe in a real implementation
            result = {'shape': df_filled.shape, 'message': 'NaN values filled'}
        elif operation == 'drop_duplicates':
            df_unique = df.drop_duplicates()
            # Note: This would modify the original dataframe in a real implementation
            result = {'shape': df_unique.shape, 'message': 'Duplicates dropped'}
        elif operation == 'sort_values':
            by = params.get('by')
            ascending = params.get('ascending', True)
            if by and by in df.columns:
                df_sorted = df.sort_values(by=by, ascending=ascending)
                # Note: This would modify the original dataframe in a real implementation
                result = {'message': f'Data sorted by {by}'}
            else:
                result = "Column not found"
        elif operation == 'reset_index':
            df_reset = df.reset_index(drop=True)
            # Note: This would modify the original dataframe in a real implementation
            result = {'message': 'Index reset'}
        elif operation == 'set_index':
            col = params.get('column')
            if col and col in df.columns:
                df_indexed = df.set_index(col)
                # Note: This would modify the original dataframe in a real implementation
                result = {'message': f'Index set to {col}'}
            else:
                result = "Column not found"
        elif operation == 'rename':
            columns = params.get('columns', {})
            df_renamed = df.rename(columns=columns)
            # Note: This would modify the original dataframe in a real implementation
            result = {'message': 'Columns renamed'}
        else:
            result = f"Operation '{operation}' not implemented"

        return {'result': result}

    except Exception as e:
        return {'error': str(e)}