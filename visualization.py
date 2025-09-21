import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import pandas as pd
import numpy as np
import io
import base64
from flask import jsonify

def create_visualization(library, chart_type, df, params=None):
    """Create visualizations using different libraries"""
    if params is None:
        params = {}

    try:
        if library == 'matplotlib':
            return create_matplotlib_chart(chart_type, df, params)
        elif library == 'seaborn':
            return create_seaborn_chart(chart_type, df, params)
        elif library == 'plotly':
            return create_plotly_chart(chart_type, df, params)
        else:
            return {'error': f'Library {library} not supported'}

    except Exception as e:
        return {'error': str(e)}

def create_matplotlib_chart(chart_type, df, params):
    """Create matplotlib charts"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == 'histogram':
            col = params.get('column')
            if col and col in df.columns:
                df[col].hist(ax=ax, bins=30, alpha=0.7, color='#667eea')
                ax.set_title(f'Histogram of {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Column not found', ha='center', va='center', transform=ax.transAxes)

        elif chart_type == 'scatter':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color='#667eea', edgecolors='white', linewidth=0.5)
                ax.set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Columns not found', ha='center', va='center', transform=ax.transAxes)

        elif chart_type == 'line':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                ax.plot(df[x_col], df[y_col], linewidth=2, color='#667eea', marker='o', markersize=4)
                ax.set_title(f'Line Plot: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Columns not found', ha='center', va='center', transform=ax.transAxes)

        elif chart_type == 'bar':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                ax.bar(df[x_col], df[y_col], color='#667eea', alpha=0.7, edgecolor='white', linewidth=0.5)
                ax.set_title(f'Bar Chart: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
            else:
                ax.text(0.5, 0.5, 'Columns not found', ha='center', va='center', transform=ax.transAxes)

        elif chart_type == 'box':
            col = params.get('column')
            if col and col in df.columns:
                ax.boxplot(df[col].dropna(), patch_artist=True,
                          boxprops=dict(facecolor='#667eea', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
                ax.set_title(f'Box Plot of {col}', fontsize=14, fontweight='bold')
                ax.set_ylabel(col, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'Column not found', ha='center', va='center', transform=ax.transAxes)

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

        return {'image': f'data:image/png;base64,{image_base64}'}

    except Exception as e:
        plt.close('all')  # Close any open figures
        return {'error': str(e)}

def create_seaborn_chart(chart_type, df, params):
    """Create seaborn charts"""
    try:
        plt.figure(figsize=(12, 8))

        if chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                # Calculate correlation matrix
                corr_matrix = numeric_df.corr()

                # Create mask for upper triangle
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

                # Create heatmap
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                           center=0, square=True, linewidths=0.5,
                           cbar_kws={"shrink": 0.8})

                plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, 'No numeric columns available', ha='center', va='center', transform=plt.gca().transAxes)

        elif chart_type == 'pairplot':
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                # Create a simplified pairplot for web display
                fig, axes = plt.subplots(numeric_df.shape[1], numeric_df.shape[1], figsize=(15, 15))
                cols = numeric_df.columns

                for i in range(len(cols)):
                    for j in range(len(cols)):
                        if i == j:
                            # Diagonal - histograms
                            axes[i, j].hist(numeric_df[cols[i]], bins=20, alpha=0.7, color='#667eea', edgecolor='white')
                            axes[i, j].set_title(cols[i], fontsize=10, fontweight='bold')
                        else:
                            # Off-diagonal - scatter plots
                            axes[i, j].scatter(numeric_df[cols[j]], numeric_df[cols[i]],
                                             alpha=0.6, color='#667eea', s=20, edgecolors='none')

                        if j == 0:
                            axes[i, j].set_ylabel(cols[i], fontsize=10)
                        if i == len(cols) - 1:
                            axes[i, j].set_xlabel(cols[j], fontsize=10)

                plt.suptitle('Pairwise Relationships', fontsize=16, fontweight='bold', y=0.95)
                plt.tight_layout()
            else:
                plt.text(0.5, 0.5, 'Need at least 2 numeric columns', ha='center', va='center', transform=plt.gca().transAxes)

        elif chart_type == 'boxplot':
            col = params.get('column')
            if col and col in df.columns:
                sns.boxplot(y=df[col], color='#667eea', width=0.5)
                plt.title(f'Box Plot of {col}', fontsize=16, fontweight='bold', pad=20)
                plt.ylabel(col, fontsize=12)
                plt.grid(True, alpha=0.3, axis='y')
            else:
                plt.text(0.5, 0.5, 'Column not found', ha='center', va='center', transform=plt.gca().transAxes)

        elif chart_type == 'violinplot':
            col = params.get('column')
            if col and col in df.columns:
                sns.violinplot(y=df[col], color='#667eea', inner='quartile')
                plt.title(f'Violin Plot of {col}', fontsize=16, fontweight='bold', pad=20)
                plt.ylabel(col, fontsize=12)
                plt.grid(True, alpha=0.3, axis='y')
            else:
                plt.text(0.5, 0.5, 'Column not found', ha='center', va='center', transform=plt.gca().transAxes)

        elif chart_type == 'histplot':
            col = params.get('column')
            if col and col in df.columns:
                sns.histplot(df[col], kde=True, color='#667eea', alpha=0.7, edgecolor='white')
                plt.title(f'Histogram of {col}', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3, axis='y')
            else:
                plt.text(0.5, 0.5, 'Column not found', ha='center', va='center', transform=plt.gca().transAxes)

        # Save plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close('all')

        return {'image': f'data:image/png;base64,{image_base64}'}

    except Exception as e:
        plt.close('all')
        return {'error': str(e)}

def create_plotly_chart(chart_type, df, params):
    """Create plotly charts"""
    try:
        if chart_type == 'scatter':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.scatter(df, x=x_col, y=y_col,
                               title=f'Scatter Plot: {x_col} vs {y_col}',
                               color_discrete_sequence=['#667eea'],
                               opacity=0.7)
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'line':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.line(df, x=x_col, y=y_col,
                            title=f'Line Plot: {x_col} vs {y_col}',
                            color_discrete_sequence=['#667eea'],
                            markers=True)
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'bar':
            x_col = params.get('x_column')
            y_col = params.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.bar(df, x=x_col, y=y_col,
                           title=f'Bar Chart: {x_col} vs {y_col}',
                           color_discrete_sequence=['#667eea'])
                fig.update_layout(
                    xaxis_title=x_col,
                    yaxis_title=y_col,
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'histogram':
            col = params.get('column')
            if col and col in df.columns:
                fig = px.histogram(df, x=col,
                                 title=f'Histogram of {col}',
                                 color_discrete_sequence=['#667eea'],
                                 opacity=0.7)
                fig.update_layout(
                    xaxis_title=col,
                    yaxis_title='Frequency',
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'box':
            col = params.get('column')
            if col and col in df.columns:
                fig = px.box(df, y=col,
                           title=f'Box Plot of {col}',
                           color_discrete_sequence=['#667eea'])
                fig.update_layout(
                    yaxis_title=col,
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig = px.imshow(numeric_df.corr(),
                              title='Correlation Heatmap',
                              color_continuous_scale='RdBu_r',
                              aspect='auto')
                fig.update_layout(
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        elif chart_type == 'pie':
            col = params.get('column')
            if col and col in df.columns:
                value_counts = df[col].value_counts().head(10)  # Limit to top 10 for readability
                fig = px.pie(values=value_counts.values,
                           names=value_counts.index,
                           title=f'Pie Chart of {col}')
                fig.update_layout(
                    font=dict(size=12),
                    title_font=dict(size=16, family='Arial', color='black')
                )

        else:
            return {'error': f'Chart type {chart_type} not supported'}

        return {'plotly_json': json.dumps(fig, cls=PlotlyJSONEncoder)}

    except Exception as e:
        return {'error': str(e)}