import numpy as np
import pandas as pd

def generate_ai_summary(df, session_id):
    """Generate comprehensive local AI-like analysis without external APIs"""
    try:
        return generate_advanced_analysis(df)
    except Exception as e:
        return {'summary': f"Analysis failed: {str(e)}\n\nBasic stats:\n- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}"}

def generate_advanced_analysis(df):
    """Generate comprehensive local AI-like analysis without external APIs"""
    try:
        analysis_parts = []

        # Executive Summary
        analysis_parts.append("🤖 INTELLIGENT DATA ANALYSIS REPORT")
        analysis_parts.append("=" * 50)
        analysis_parts.append("")

        # Dataset Overview with Intelligence
        analysis_parts.append("📊 DATASET OVERVIEW")
        analysis_parts.append("-" * 30)
        analysis_parts.append(f"• Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns")
        analysis_parts.append(f"• Memory footprint: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        # Size classification with insights
        if df.shape[0] < 100:
            size_insight = "Small dataset - good for detailed analysis, may have limited statistical power"
        elif df.shape[0] < 1000:
            size_insight = "Medium-small dataset - suitable for most analyses with careful validation"
        elif df.shape[0] < 10000:
            size_insight = "Medium dataset - good balance of statistical power and computational efficiency"
        elif df.shape[0] < 100000:
            size_insight = "Large dataset - may require sampling for some analyses, good for robust insights"
        else:
            size_insight = "Very large dataset - consider distributed computing approaches"
        analysis_parts.append(f"• Size Assessment: {size_insight}")
        analysis_parts.append("")

        # Data Quality Intelligence
        analysis_parts.append("🔍 DATA QUALITY ASSESSMENT")
        analysis_parts.append("-" * 30)

        missing_total = df.isnull().sum().sum()
        missing_percentage = (missing_total / (df.shape[0] * df.shape[1])) * 100

        if missing_percentage == 0:
            quality_status = "Excellent - No missing data detected"
        elif missing_percentage < 1:
            quality_status = "Very Good - Minimal missing data (<1%)"
        elif missing_percentage < 5:
            quality_status = "Good - Moderate missing data (1-5%)"
        elif missing_percentage < 15:
            quality_status = "Fair - Significant missing data (5-15%)"
        else:
            quality_status = "Poor - Extensive missing data (>15%)"

        analysis_parts.append(f"• Overall Quality: {quality_status}")
        analysis_parts.append(f"• Missing Values: {missing_total:,} total ({missing_percentage:.2f}%)")

        # Column-specific missing data insights
        missing_cols = df.isnull().sum()
        problematic_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        if len(problematic_cols) > 0:
            analysis_parts.append("• Columns with missing data:")
            for col, count in problematic_cols.head(5).items():
                percentage = (count / df.shape[0]) * 100
                severity = "Critical" if percentage > 50 else "High" if percentage > 20 else "Moderate" if percentage > 5 else "Low"
                analysis_parts.append(f"  - {col}: {count:,} missing ({percentage:.1f}%) - {severity} concern")
        analysis_parts.append("")

        # Data Type Intelligence
        analysis_parts.append("🏷️  DATA TYPE ANALYSIS")
        analysis_parts.append("-" * 30)

        dtypes_count = df.dtypes.value_counts()
        for dtype, count in dtypes_count.items():
            analysis_parts.append(f"• {dtype}: {count} columns")

        # Smart column classification
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        boolean_cols = df.select_dtypes(include=['bool']).columns

        analysis_parts.append("• Column Classification:")
        analysis_parts.append(f"  - Numeric: {len(numeric_cols)} columns")
        analysis_parts.append(f"  - Categorical: {len(categorical_cols)} columns")
        analysis_parts.append(f"  - DateTime: {len(datetime_cols)} columns")
        analysis_parts.append(f"  - Boolean: {len(boolean_cols)} columns")
        analysis_parts.append("")

        # Statistical Intelligence for Numeric Data
        if len(numeric_cols) > 0:
            analysis_parts.append("📈 NUMERIC DATA INSIGHTS")
            analysis_parts.append("-" * 30)

            # Overall numeric statistics
            numeric_stats = df[numeric_cols].describe()
            analysis_parts.append("• Summary Statistics:")
            analysis_parts.append(f"  - Total numeric variables: {len(numeric_cols)}")
            analysis_parts.append(f"  - Average values per column: {numeric_stats.loc['mean'].mean():.2f}")
            analysis_parts.append(f"  - Data variability (avg std): {numeric_stats.loc['std'].mean():.2f}")

            # Distribution insights
            skewness = df[numeric_cols].skew()
            highly_skewed = skewness[abs(skewness) > 1]
            if len(highly_skewed) > 0:
                analysis_parts.append("• Distribution Insights:")
                for col, skew_val in highly_skewed.items():
                    direction = "right-skewed" if skew_val > 0 else "left-skewed"
                    analysis_parts.append(f"  - {col}: Highly {direction} (skewness: {skew_val:.2f})")

            # Outlier detection insights
            analysis_parts.append("• Outlier Analysis:")
            for col in numeric_cols[:5]:  # Analyze first 5 numeric columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = (outliers / df.shape[0]) * 100
                if outlier_percentage > 5:
                    analysis_parts.append(f"  - {col}: {outliers:,} potential outliers ({outlier_percentage:.1f}%)")
            analysis_parts.append("")

        # Categorical Data Intelligence
        if len(categorical_cols) > 0:
            analysis_parts.append("📂 CATEGORICAL DATA INSIGHTS")
            analysis_parts.append("-" * 30)

            analysis_parts.append(f"• Categorical variables: {len(categorical_cols)}")
            for col in categorical_cols[:5]:  # Analyze first 5 categorical columns
                try:
                    unique_count = df[col].nunique()
                    most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                    most_common_count = df[col].value_counts().iloc[0] if unique_count > 0 else 0
                    most_common_percentage = (most_common_count / df.shape[0]) * 100

                    analysis_parts.append(f"• {col}:")
                    analysis_parts.append(f"  - Unique values: {unique_count}")
                    analysis_parts.append(f"  - Most common: '{most_common}' ({most_common_count:,} occurrences, {most_common_percentage:.1f}%)")

                    if unique_count > 20:
                        analysis_parts.append("  - High cardinality - consider grouping or encoding strategies")
                    elif unique_count == 1:
                        analysis_parts.append("  - Single unique value - may not be informative")
                except:
                    analysis_parts.append(f"- {col}: Analysis unavailable")
            analysis_parts.append("")

        # Correlation Intelligence
        if len(numeric_cols) > 1:
            analysis_parts.append("🔗 CORRELATION ANALYSIS")
            analysis_parts.append("-" * 30)

            try:
                corr_matrix = df[numeric_cols].corr()

                # Find strongest correlations
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val):
                            corr_pairs.append((numeric_cols[i], numeric_cols[j], abs(corr_val), corr_val))

                if corr_pairs:
                    corr_pairs.sort(key=lambda x: x[2], reverse=True)

                    analysis_parts.append("• Strongest Relationships:")
                    for col1, col2, abs_corr, corr in corr_pairs[:5]:
                        direction = "positive" if corr > 0 else "negative"
                        strength = "Very Strong" if abs_corr > 0.8 else "Strong" if abs_corr > 0.6 else "Moderate" if abs_corr > 0.4 else "Weak"
                        analysis_parts.append(f"  - {col1} ↔ {col2}: {strength} {direction} ({corr:.3f})")

                    # Multicollinearity warning
                    strong_corr_count = sum(1 for _, _, abs_corr, _ in corr_pairs if abs_corr > 0.8)
                    if strong_corr_count > 0:
                        analysis_parts.append(f"• ⚠️  Multicollinearity Alert: {strong_corr_count} very strong correlations detected")
                        analysis_parts.append("  - Consider dimensionality reduction or feature selection")

            except Exception as e:
                analysis_parts.append(f"• Correlation analysis failed: {str(e)}")
            analysis_parts.append("")

        # Analytics Readiness Assessment
        analysis_parts.append("🎯 ANALYTICS READINESS ASSESSMENT")
        analysis_parts.append("-" * 30)

        readiness_score = 0
        readiness_factors = []

        # Size factor
        if df.shape[0] >= 100:
            readiness_score += 2
            readiness_factors.append("✓ Sufficient sample size")
        elif df.shape[0] >= 30:
            readiness_score += 1
            readiness_factors.append("⚠️  Small sample - limited statistical power")

        # Quality factor
        if missing_percentage < 5:
            readiness_score += 2
            readiness_factors.append("✓ Good data quality")
        elif missing_percentage < 15:
            readiness_score += 1
            readiness_factors.append("⚠️  Moderate data quality issues")

        # Feature factor
        if len(numeric_cols) >= 3:
            readiness_score += 2
            readiness_factors.append("✓ Sufficient numeric features")
        elif len(numeric_cols) >= 1:
            readiness_score += 1
            readiness_factors.append("⚠️  Limited numeric features")

        # Diversity factor
        total_features = len(numeric_cols) + len(categorical_cols)
        if total_features >= 5:
            readiness_score += 1
            readiness_factors.append("✓ Feature diversity present")

        # Scoring
        max_score = 6
        readiness_percentage = (readiness_score / max_score) * 100

        if readiness_percentage >= 80:
            readiness_level = "Excellent - Ready for advanced analytics"
        elif readiness_percentage >= 60:
            readiness_level = "Good - Suitable for most analyses"
        elif readiness_percentage >= 40:
            readiness_level = "Fair - May require data preprocessing"
        else:
            readiness_level = "Poor - Significant data preparation needed"

        analysis_parts.append(f"• Analytics Readiness: {readiness_level} ({readiness_percentage:.0f}%)")
        for factor in readiness_factors:
            analysis_parts.append(f"  {factor}")
        analysis_parts.append("")

        # Recommendations
        analysis_parts.append("💡 RECOMMENDATIONS")
        analysis_parts.append("-" * 30)

        recommendations = []

        if missing_percentage > 10:
            recommendations.append("• Data Cleaning: Address missing values through imputation or removal")
        if len(numeric_cols) > 0 and len(highly_skewed) > 0:
            recommendations.append("• Transformation: Consider log/sqrt transformations for skewed variables")
        if len(categorical_cols) > 0:
            recommendations.append("• Encoding: Convert categorical variables for ML (OneHot, Label Encoding)")
        if len(numeric_cols) > 1 and strong_corr_count > 2:
            recommendations.append("• Feature Selection: Remove highly correlated features to reduce multicollinearity")
        if df.shape[0] < 100:
            recommendations.append("• Data Collection: Consider gathering more samples for robust analysis")
        if len(numeric_cols) == 0:
            recommendations.append("• Feature Engineering: Create numeric features from existing data")

        if not recommendations:
            recommendations.append("• Dataset looks well-prepared for analysis!")

        for rec in recommendations:
            analysis_parts.append(rec)
        analysis_parts.append("")

        # Potential Use Cases
        analysis_parts.append("🚀 POTENTIAL USE CASES")
        analysis_parts.append("-" * 30)

        use_cases = []

        if len(numeric_cols) >= 2:
            use_cases.append("• Predictive Modeling: Regression or classification tasks")
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            use_cases.append("• Customer Segmentation: Clustering analysis")
        if len(datetime_cols) > 0:
            use_cases.append("• Time Series Analysis: Trend and seasonality detection")
        if len(categorical_cols) >= 2:
            use_cases.append("• Association Rule Mining: Market basket analysis")
        if df.shape[0] > 1000:
            use_cases.append("• Statistical Testing: Hypothesis testing and A/B analysis")

        if not use_cases:
            use_cases.append("• Exploratory Data Analysis: Understanding data patterns and distributions")

        for use_case in use_cases:
            analysis_parts.append(use_case)

        return {'summary': "\n".join(analysis_parts)}

    except Exception as e:
        return {'summary': f"Advanced analysis failed: {str(e)}\n\nBasic stats:\n- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}"}