# 📊 OpenStatica

<div align="center">
  <img src="frontend/assets/logo.svg" alt="OpenStatica Logo" width="200"/>
  
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
  [![React](https://img.shields.io/badge/React-18%2B-blue)](https://reactjs.org)
</div>

<h3 align="center">🚀 Open-Source Statistical Analysis & Machine Learning Platform</h3>

<p align="center">
  <strong>Transform your data into insights with powerful statistics, beautiful visualizations, and cutting-edge ML - all in your browser!</strong>
</p>

---

## ✨ What's New in v1.0.0

- 🎨 **Modern UI/UX**: Beautiful, responsive design with dark mode support
- 📱 **Mobile Friendly**: Fully responsive interface that works on any device
- 🧪 **Complete Statistical Tests**: T-tests, ANOVA, Chi-square, Regression, and more
- 🤖 **Machine Learning Suite**: Classification, Regression, Clustering, Deep Learning
- 📊 **Advanced Visualizations**: Interactive charts with Plotly
- 🔄 **Data Transformation**: Comprehensive data preprocessing tools
- 💾 **Export & Share**: Multiple export formats and collaboration features
- 🔌 **Plugin System**: Extensible architecture for custom additions

---

## 🎯 Features

### 📁 Data Management
- **Multi-format Support**: CSV, Excel, JSON, Parquet, SPSS, Stata, SAS
- **Data Quality Assessment**: Automatic profiling and quality scoring
- **Smart Type Detection**: Automatic detection of numeric, categorical, datetime columns
- **Large Dataset Support**: Efficient handling of big data with streaming
- **Missing Value Analysis**: Comprehensive missing data reports

### 📊 Statistical Analysis

#### Descriptive Statistics
- Mean, Median, Mode, Standard Deviation
- Quartiles, IQR, Range
- Skewness, Kurtosis
- Confidence Intervals
- Outlier Detection
- Distribution Analysis

#### Inferential Statistics
- **T-Tests**: One-sample, Independent, Paired
- **ANOVA**: One-way, Two-way, Repeated Measures, MANOVA
- **Chi-Square Tests**: Independence, Goodness of Fit
- **Correlation**: Pearson, Spearman, Kendall, Partial
- **Regression**: Linear, Multiple, Polynomial, Logistic, Ridge, Lasso
- **Non-parametric**: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman

### 🤖 Machine Learning

#### Supervised Learning
- **Classification**: Random Forest, SVM, XGBoost, Neural Networks
- **Regression**: Linear, Ridge, Lasso, Elastic Net, SVR, Gradient Boosting
- **Model Evaluation**: Cross-validation, ROC curves, Feature Importance
- **AutoML**: Automatic model selection and hyperparameter tuning

#### Unsupervised Learning
- **Clustering**: K-Means, DBSCAN, Hierarchical, Gaussian Mixture
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, LDA, ICA
- **Anomaly Detection**: Isolation Forest, LOF, One-Class SVM

#### Deep Learning (Optional)
- Multi-layer Perceptrons
- Convolutional Neural Networks
- Recurrent Neural Networks (LSTM, GRU)
- Transformers
- AutoEncoders

### 📈 Visualizations
- **Basic Charts**: Histogram, Bar, Line, Scatter, Box Plot
- **Advanced**: Heatmaps, 3D Plots, Contour, Violin Plots
- **Statistical**: Q-Q Plots, Residual Plots, ROC Curves
- **Interactive**: Zoom, Pan, Hover details, Export
- **Customizable**: Themes, Colors, Annotations

### 🔄 Data Transformation
- **Normalization**: Z-score, Min-Max, Robust Scaling
- **Encoding**: One-hot, Label, Ordinal, Target
- **Imputation**: Mean, Median, Mode, Forward/Backward Fill, Interpolation
- **Feature Engineering**: Polynomial Features, Interactions, Binning
- **Time Series**: Differencing, Lag Features, Rolling Statistics

### 💾 Export & Collaboration
- **Export Formats**: CSV, Excel, JSON, PDF, HTML Reports
- **Share Results**: Secure sharing links
- **Session Management**: Save and restore analysis sessions
- **Report Generation**: Automated report creation with visualizations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shamspias/open-statica.git
cd open-statica
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd ../frontend
npm install
```

4. **Environment Configuration**
```bash
# In backend directory
cp .env.example .env
# Edit .env with your settings
```

5. **Run the Application**

Backend:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm start
```

6. **Access the Application**
Open your browser and navigate to `http://localhost:3000`


---

## 📖 Usage Guide

### 1. Upload Data
- Click on "Upload" in the sidebar
- Drag & drop or select your data file
- Automatic data profiling and quality assessment

### 2. Explore Data
- View data tables with sorting and filtering
- Check data quality scores
- Identify missing values and outliers

### 3. Transform Data (Optional)
- Apply normalization or standardization
- Handle missing values
- Create new features

### 4. Statistical Analysis
- **Descriptive**: Get summary statistics for all variables
- **Tests**: Run hypothesis tests with automatic assumption checking
- **Regression**: Build predictive models with diagnostics

### 5. Machine Learning
- Select task type (Classification/Regression/Clustering)
- Choose algorithm
- Configure parameters
- Train and evaluate models
- Deploy for predictions

### 6. Visualize Results
- Create interactive charts
- Customize appearance
- Export as images or interactive HTML

### 7. Export & Share
- Download results in multiple formats
- Generate comprehensive reports
- Share analysis via secure links

---

### Technology Stack

**Backend:**
- **Framework**: FastAPI (async Python web framework)
- **Data Processing**: Pandas, Polars, NumPy
- **Statistics**: SciPy, Statsmodels, Pingouin
- **Machine Learning**: Scikit-learn, XGBoost, (Optional: TensorFlow/PyTorch)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Database**: PostgreSQL (optional), Redis (caching)

**Frontend:**
- **Framework**: React 18
- **UI Library**: Custom modern design system
- **Charts**: Plotly.js
- **State Management**: React Hooks
- **HTTP Client**: Fetch API

---


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📈 Roadmap

### Version 1.1 (Q1 2025)
- [ ] Real-time collaboration
- [ ] Advanced time series analysis
- [ ] More deep learning models
- [ ] Automated reporting

### Version 1.2 (Q2 2025)
- [ ] Cloud deployment options
- [ ] Multi-language support
- [ ] Advanced AutoML features
- [ ] Integration with data lakes

### Version 2.0 (Q3 2025)
- [ ] Distributed computing support
- [ ] Advanced NLP capabilities
- [ ] Computer vision features
- [ ] Enterprise features (SSO, audit logs)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- FastAPI for the amazing web framework
- React team for the frontend library
- Scikit-learn for ML algorithms
- Plotly for interactive visualizations
- All contributors and users of OpenStatica

---

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/shamspias/openstatica/issues)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=shamspias/open-statica&type=Date)](https://star-history.com/#shamspias/open-statica&Date)

---

<div align="center">
  <strong>by The Wandering Algorithm</strong>
  <br>
  <sub>Making statistical analysis accessible to everyone</sub>
</div>