import io
import base64
import tempfile
import logging
from flask import send_file

# Report generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. PDF reports will be disabled.")

# HTML report generation
from jinja2 import Template

logger = logging.getLogger(__name__)

class ReportGeneratorService:
    """Service for generating data analysis reports"""

    def __init__(self):
        self.report_store = {}  # Store generated reports
        self.report_timestamps = {}  # Track report access times
        self.MAX_REPORTS = 50
        self.REPORT_TIMEOUT = 86400  # 24 hours

    def generate_visualization_base64(self, df, viz_type, params):
        """Generate visualization as base64 encoded image"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np

            plt.figure(figsize=(8, 6))

            if viz_type == 'histogram':
                col = params.get('column')
                if col and col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].hist(bins=30, alpha=0.7)
                        plt.title(f'Distribution of {col}')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')
                    else:
                        df[col].value_counts().head(10).plot(kind='bar')
                        plt.title(f'Top 10 values in {col}')
                        plt.xticks(rotation=45)
            elif viz_type == 'correlation_heatmap':
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty and numeric_df.shape[1] > 1:
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
                    plt.title('Correlation Matrix')
            elif viz_type == 'scatter':
                x_col = params.get('x_column')
                y_col = params.get('y_column')
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    plt.scatter(df[x_col], df[y_col], alpha=0.6)
                    plt.title(f'{x_col} vs {y_col}')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
            elif viz_type == 'boxplot':
                col = params.get('column')
                if col and col in df.columns and df[col].dtype in ['int64', 'float64']:
                    plt.boxplot(df[col].dropna())
                    plt.title(f'Box Plot of {col}')
                    plt.ylabel(col)

            plt.tight_layout()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            return image_base64
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return None

    def generate_pdf_report(self, df, title, include_stats=True, include_correlations=True,
                           include_visualizations=True, include_ai=True, ai_insights=""):
        """Generate PDF report using ReportLab"""
        if not REPORTLAB_AVAILABLE:
            raise Exception("ReportLab not available for PDF generation")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 12))

        # Dataset overview
        story.append(Paragraph("Dataset Overview", styles['Heading2']))
        overview_data = [
            ["Rows", str(df.shape[0])],
            ["Columns", str(df.shape[1])],
            ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"]
        ]
        overview_table = Table(overview_data, colWidths=[2*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 12))

        # Statistics
        if include_stats:
            story.append(Paragraph("Statistical Summary", styles['Heading2']))
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats_data = [['Column', 'Count', 'Mean', 'Std', 'Min', 'Max']]
                for col in numeric_cols[:10]:  # Limit to first 10 columns
                    desc = df[col].describe()
                    stats_data.append([
                        col,
                        f"{desc['count']:.0f}",
                        f"{desc['mean']:.2f}",
                        f"{desc['std']:.2f}",
                        f"{desc['min']:.2f}",
                        f"{desc['max']:.2f}"
                    ])

                stats_table = Table(stats_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(stats_table)
            story.append(Spacer(1, 12))

        # Visualizations
        if include_visualizations:
            story.append(Paragraph("Visualizations", styles['Heading2']))

            # Histogram
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                story.append(Paragraph("Distribution Histogram", styles['Heading3']))
                img_data = self.generate_visualization_base64(df, 'histogram', {'column': numeric_cols[0]})
                if img_data:
                    img_buffer = io.BytesIO(base64.b64decode(img_data))
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                story.append(Spacer(1, 12))

            # Correlation heatmap
            if len(numeric_cols) > 1:
                story.append(Paragraph("Correlation Matrix", styles['Heading3']))
                img_data = self.generate_visualization_base64(df, 'correlation_heatmap', {})
                if img_data:
                    img_buffer = io.BytesIO(base64.b64decode(img_data))
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                story.append(Spacer(1, 12))

        # AI Insights
        if include_ai and ai_insights:
            story.append(Paragraph("AI-Generated Insights", styles['Heading2']))
            story.append(Paragraph(ai_insights, styles['Normal']))
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)
        return buffer

    def generate_html_report(self, df, title, include_stats=True, include_correlations=True,
                            include_visualizations=True, include_ai=True, ai_insights=""):
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .stats-table { font-size: 12px; }
                .visualization { margin: 20px 0; text-align: center; }
                .visualization img { max-width: 100%; height: auto; }
                .insights { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>

            <h2>Dataset Overview</h2>
            <table>
                <tr><th>Rows</th><td>{{ df_shape[0] }}</td></tr>
                <tr><th>Columns</th><td>{{ df_shape[1] }}</td></tr>
                <tr><th>Memory Usage</th><td>{{ "%.2f"|format(df.memory_usage(deep=True).sum() / 1024 / 1024) }} MB</td></tr>
            </table>

            {% if include_stats %}
            <h2>Statistical Summary</h2>
            {% if numeric_cols %}
            <table class="stats-table">
                <tr>
                    <th>Column</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
                {% for col in numeric_cols[:10] %}
                {% set desc = df[col].describe() %}
                <tr>
                    <td>{{ col }}</td>
                    <td>{{ "%.0f"|format(desc['count']) }}</td>
                    <td>{{ "%.2f"|format(desc['mean']) }}</td>
                    <td>{{ "%.2f"|format(desc['std']) }}</td>
                    <td>{{ "%.2f"|format(desc['min']) }}</td>
                    <td>{{ "%.2f"|format(desc['max']) }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
            {% endif %}

            {% if include_visualizations %}
            <h2>Visualizations</h2>

            {% if numeric_cols %}
            <div class="visualization">
                <h3>Distribution Histogram</h3>
                <img src="data:image/png;base64,{{ generate_visualization_base64(df, 'histogram', {'column': numeric_cols[0]}) }}" alt="Histogram">
            </div>
            {% endif %}

            {% if numeric_cols|length > 1 %}
            <div class="visualization">
                <h3>Correlation Matrix</h3>
                <img src="data:image/png;base64,{{ generate_visualization_base64(df, 'correlation_heatmap', {}) }}" alt="Correlation Heatmap">
            </div>
            {% endif %}
            {% endif %}

            {% if include_ai and ai_insights %}
            <h2>AI-Generated Insights</h2>
            <div class="insights">
                {{ ai_insights }}
            </div>
            {% endif %}
        </body>
        </html>
        """

        # Convert pandas objects to Python types for Jinja2
        numeric_cols_list = list(df.select_dtypes(include=['number']).columns)
        df_shape = df.shape

        template = Template(html_template)
        html_content = template.render(
            title=title,
            df=df,
            df_shape=df_shape,
            numeric_cols=numeric_cols_list,
            include_stats=include_stats,
            include_visualizations=include_visualizations,
            include_ai=include_ai,
            ai_insights=ai_insights,
            generate_visualization_base64=self.generate_visualization_base64
        )

        return html_content

    def create_report(self, df, report_type, title, include_stats=True, include_correlations=True,
                     include_visualizations=True, include_ai=True, ai_insights=""):
        """Create a report and return the content"""
        try:
            if report_type == 'pdf':
                if not REPORTLAB_AVAILABLE:
                    return {'error': 'PDF generation not available. Install ReportLab.'}

                report_buffer = self.generate_pdf_report(
                    df, title, include_stats, include_correlations,
                    include_visualizations, include_ai, ai_insights
                )
                report_content = report_buffer.getvalue()
                filename = f"{title.replace(' ', '_')}.pdf"
                content_type = 'application/pdf'

            elif report_type == 'html':
                report_content = self.generate_html_report(
                    df, title, include_stats, include_correlations,
                    include_visualizations, include_ai, ai_insights
                ).encode('utf-8')
                filename = f"{title.replace(' ', '_')}.html"
                content_type = 'text/html'

            else:
                return {'error': f'Unsupported report type: {report_type}'}

            # Generate unique report ID
            import uuid
            report_id = str(uuid.uuid4)

            # Store report
            self.report_store[report_id] = {
                'content': report_content,
                'filename': filename,
                'content_type': content_type,
                'created_at': __import__('time').time()
            }
            self.report_timestamps[report_id] = __import__('time').time()

            # Clean up old reports if needed
            self._cleanup_expired_reports()

            logger.info(f"Successfully created {report_type} report: {report_id}")

            return {
                'report_id': report_id,
                'filename': filename,
                'report_type': report_type,
                'message': f'Report created successfully'
            }

        except Exception as e:
            logger.error(f"Create report error: {str(e)}")
            return {'error': 'Internal server error during report creation'}

    def get_report(self, report_id):
        """Retrieve a stored report"""
        try:
            if report_id not in self.report_store:
                return {'error': 'Report not found'}

            report_data = self.report_store[report_id]
            self.report_timestamps[report_id] = __import__('time').time()  # Update access time

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(report_data['content'])
                temp_file = f.name

            return send_file(
                temp_file,
                as_attachment=True,
                download_name=report_data['filename'],
                mimetype=report_data['content_type']
            )

        except Exception as e:
            logger.error(f"Download report error: {str(e)}")
            return {'error': 'Internal server error during report download'}

    def _cleanup_expired_reports(self):
        """Clean up expired reports"""
        current_time = __import__('time').time()
        expired_reports = []

        for report_id, timestamp in self.report_timestamps.items():
            if current_time - timestamp > self.REPORT_TIMEOUT:
                expired_reports.append(report_id)

        for report_id in expired_reports:
            if report_id in self.report_store:
                del self.report_store[report_id]
            if report_id in self.report_timestamps:
                del self.report_timestamps[report_id]
            logger.info(f"Cleaned up expired report: {report_id}")

        # Limit total reports
        if len(self.report_store) > self.MAX_REPORTS:
            sorted_reports = sorted(self.report_timestamps.items(), key=lambda x: x[1])
            reports_to_remove = len(self.report_store) - self.MAX_REPORTS

            for report_id, _ in sorted_reports[:reports_to_remove]:
                if report_id in self.report_store:
                    del self.report_store[report_id]
                if report_id in self.report_timestamps:
                    del self.report_timestamps[report_id]
                logger.info(f"Removed old report due to limit: {report_id}")

# Global instance
report_generator_service = ReportGeneratorService()