# 📊 Data Hub - Comprehensive Data Analysis Platform

A powerful web application built with Flask and HTML/CSS/JS that provides comprehensive access to Python's data science libraries including pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, and TensorFlow, along with AI-powered insights using OpenAI GPT.

## 🚀 Features

### Core Functionality
- **File Upload**: Support for CSV and JSON file formats
- **Data Preview**: Interactive data exploration with basic statistics
- **Comprehensive Library Access**: Full-featured interfaces for all major Python data science libraries

### Library Integrations

#### 🐼 Pandas Operations
- Data manipulation and cleaning
- Statistical analysis
- Data transformation operations
- Searchable function access

#### 🔢 NumPy Operations
- Array operations and mathematical functions
- Linear algebra operations
- Statistical computations
- Searchable function access

#### 📈 Data Visualization
- **Matplotlib**: Static plots and charts
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive web-based visualizations
- Multiple chart types: histograms, scatter plots, line plots, bar charts, box plots, heatmaps, etc.

#### 🤖 Machine Learning (Scikit-learn)
- Linear and non-linear regression models
- Classification algorithms
- Clustering techniques
- Model evaluation metrics
- Searchable algorithm access

#### 🧠 Deep Learning (TensorFlow)
- Neural network architectures
- CNN and RNN models
- Custom model training
- Performance visualization

#### 🤖 AI-Powered Insights
- Automated statistical analysis using OpenAI GPT
- Intelligent data insights and recommendations
- Natural language explanations

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install flask flask-cors pandas numpy matplotlib seaborn plotly scikit-learn tensorflow openai python-dotenv
   ```

3. **Set up environment variables**:
   - The `.env` file should contain your OpenAI API key:
   ```
   OPEN_API_KEY=your_openai_api_key_here
   ```

## 🚀 Running the Application

1. **Navigate to the project directory**:
   ```bash
   cd path/to/datahub
   ```

2. **Run the Flask backend**:
   ```bash
   python backend.py
   ```

3. **Open your browser** and go to `http://localhost:8000` (frontend) and `http://localhost:5000` (backend API)

## 📖 Usage Guide

### Getting Started
1. **Upload Data**: Use the "Data Upload" page to upload CSV or JSON files
2. **Explore Data**: View data preview and basic statistics
3. **Navigate**: Use the sidebar to access different analysis tools

### Library-Specific Usage

#### Pandas Operations
- Browse available operations or search for specific functions
- Click buttons to execute operations on your data
- View results in tabular format

#### NumPy Operations
- Works with numeric columns from your dataset
- Access mathematical and array operations
- Results displayed as arrays or dataframes

#### Visualization
- Select visualization type from dropdown
- Choose appropriate columns for your chart
- Interactive Plotly charts for web-based exploration

#### Machine Learning
- Select target variable for supervised learning
- Choose from various algorithms
- View model performance metrics and predictions

#### Deep Learning
- Configure neural network architectures
- Train models on your data
- Monitor training progress and results

#### AI Summary
- Generate intelligent insights about your dataset
- Get recommendations and statistical analysis
- Powered by OpenAI GPT

## 🔧 Configuration

### Environment Variables
- `OPEN_API_KEY`: Your OpenAI API key for AI summary generation
- `USE_AI`: Set to `true` to enable AI features, `false` to disable (default: `true`)

### Data Requirements
- **CSV Files**: Standard comma-separated values
- **JSON Files**: Valid JSON format
- **Data Types**: Mixed data types supported
- **Size Limits**: Depends on system memory

## 🏗️ Architecture

- **Frontend**: HTML/CSS/JavaScript web interface
- **Backend**: Flask API with data science libraries
- **AI Integration**: OpenAI GPT API
- **Data Storage**: In-memory session storage
- **Visualization**: Multiple plotting libraries (Matplotlib, Seaborn, Plotly)

## 📋 Requirements

- Python 3.7+
- Internet connection for AI features
- OpenAI API key
- Sufficient RAM for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Troubleshooting

### Common Issues
- **Flask not found**: Use `python backend.py`
- **API Key errors**: Ensure `.env` file has valid OpenAI API key
- **API Quota Exceeded**: The system will automatically provide basic analysis instead
  - Check your OpenAI billing: https://platform.openai.com/usage
  - Add credits to your account or use a different API key
  - Set `USE_AI=false` in `.env` for offline mode
- **Memory errors**: Reduce dataset size or increase system RAM
- **Import errors**: Ensure all dependencies are installed
- **AI Summary not working**: Check OpenAI API key validity and quota

### Support
- Check the terminal output for error messages
- Verify all dependencies are installed
- Ensure your data files are properly formatted

---

**Built with ❤️ using Flask, HTML/CSS/JS and Python's data science ecosystem**