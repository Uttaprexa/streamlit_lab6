# Heart Disease Prediction Dashboard

**Interactive Streamlit Dashboard for ML Model Analysis and Real-time Predictions**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![MLflow](https://img.shields.io/badge/MLflow-2.8-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Dataset](#dataset)
- [Models](#models)
- [Docker Deployment](#docker-deployment)
- [Modifications](#modifications)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This interactive Streamlit dashboard provides a comprehensive interface for:
- **Comparing** 7 different ML models for heart disease prediction
- **Making** real-time predictions with adjustable patient parameters
- **Exploring** MLflow experiment tracking data
- **Analyzing** the heart disease dataset with interactive visualizations

**Built for:** MLOps Lab Assignment  
**Course:** Machine Learning Operations, Northeastern University  
**Semester:** Fall 2025

---

## Features

### Home Page
- Dashboard overview with key statistics
- Quick access to all pages
- Dataset distribution visualization
- Interactive navigation

### Model Comparison
- Compare 7 ML models side-by-side
- Interactive performance charts (bar, line, scatter)
- ROC curve analysis
- Confusion matrices for each model
- Detailed classification reports
- Model ranking system

### Live Prediction
- Real-time heart disease risk prediction
- Interactive sliders for patient parameters
- Multiple model selection
- Probability gauge visualization
- Risk interpretation and recommendations
- Export prediction results

### MLflow Explorer
- View all experiment runs
- Compare hyperparameters
- Analyze training metrics
- Export experiment data
- Timeline visualization

### Data Analysis
- Comprehensive EDA (Exploratory Data Analysis)
- Feature distribution analysis
- Correlation heatmaps
- Pairwise relationship exploration
- Data quality reports
- Interactive filtering and visualization

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/Uttaprexa/streamlit_lab6.git
cd streamlit-ml-dashboard

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Generate sample data (if needed)
python scripts/generate_data.py

# 6. Run the dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## Usage

### Running the Dashboard
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Run Streamlit
streamlit run app.py
```

### Navigation

The dashboard has 5 main pages accessible via the sidebar:

1. **Home** - Overview and quick stats
2. **Model Comparison** - Compare all ML models
3. **Live Prediction** - Make real-time predictions
4. **MLflow Explorer** - View experiment tracking
5. **Data Analysis** - Explore the dataset

### Making Predictions

1. Navigate to "Live Prediction" page
2. Select a model from the dropdown
3. Adjust patient parameters using sliders
4. Click "PREDICT RISK"
5. View results and recommendations

---

## Project Structure
```
streamlit-ml-dashboard/
├── app.py                          # Main home page
├── pages/                          # Multi-page app
│   ├── 1_Model_Comparison.py       # Model comparison page
│   ├── 2_Live_Prediction.py        # Prediction page
│   ├── 3_MLflow_Explorer.py        # MLflow tracking page
│   └── 4_Data_Analysis.py          # Data analysis page
├── utils/                          # Helper functions
│   ├── __init__.py
│   ├── mlflow_loader.py           # MLflow utilities
│   ├── data_utils.py              # Data processing
│   └── visualizations.py          # Plotting functions
├── data/
│   └── heart_disease.csv          # Dataset
├── mlruns/                        # MLflow tracking data
├── models/                        # Saved models
├── outputs/                       # Generated plots
├── scripts/
│   └── generate_data.py           # Data generation
├── .streamlit/
│   └── config.toml                # Streamlit config
├── Dockerfile                     # Docker container
├── docker-compose.yml             # Docker Compose config
├── requirements.txt               # Python dependencies
├── .gitignore
├── .dockerignore
└── README.md                      # This file
```

---

## Technologies

### Core Framework
- **Streamlit 1.28+** - Dashboard framework
- **Python 3.11** - Programming language

### Machine Learning
- **scikit-learn 1.3+** - ML algorithms
- **MLflow 2.8+** - Experiment tracking
- **pandas 2.0+** - Data manipulation
- **numpy 1.24+** - Numerical computing

### Visualization
- **Plotly 5.17+** - Interactive charts
- **Matplotlib 3.7+** - Static plots
- **Seaborn 0.12+** - Statistical visualization

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

---

## Dataset

### Heart Disease Dataset

**Source:** Synthetic dataset generated for this project  
**Purpose:** Educational/demonstration

**Specifications:**
- **Total Samples:** 500 patients
- **Features:** 13 clinical parameters
- **Target:** Binary (0 = No Disease, 1 = Disease)

**Features:**

| Feature | Description | Range |
|---------|-------------|-------|
| age | Age in years | 30-80 |
| sex | Sex (0=Female, 1=Male) | 0-1 |
| cp | Chest pain type | 0-3 |
| trestbps | Resting blood pressure (mm Hg) | 90-200 |
| chol | Serum cholesterol (mg/dl) | 120-400 |
| fbs | Fasting blood sugar > 120 mg/dl | 0-1 |
| restecg | Resting ECG results | 0-2 |
| thalach | Maximum heart rate achieved | 70-200 |
| exang | Exercise induced angina | 0-1 |
| oldpeak | ST depression | 0-6 |
| slope | Slope of peak exercise ST segment | 0-2 |
| ca | Number of major vessels | 0-3 |
| thal | Thalassemia | 0-3 |

**Target Distribution:**
- Class 0 (No Disease): ~60%
- Class 1 (Disease): ~40%

---

## Models

The dashboard includes 7 trained ML models:

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Gradient Boosting** | **95.00%** | **95.20%** | **95.00%** | **95.10%** | 0.15s |
| Random Forest | 93.00% | 93.15% | 93.00% | 93.08% | 0.16s |
| SVM | 88.00% | 88.20% | 88.00% | 88.10% | 0.02s |
| Logistic Regression | 85.00% | 85.28% | 85.00% | 85.11% | 0.04s |
| KNN | 83.00% | 83.15% | 83.00% | 83.08% | 0.01s |
| Decision Tree | 80.00% | 80.25% | 80.00% | 80.12% | 0.02s |
| Naive Bayes | 78.00% | 78.10% | 78.00% | 78.05% | 0.01s |

### Model Descriptions

1. **Logistic Regression** - Linear classifier for baseline
2. **Decision Tree** - Tree-based model for interpretability
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble (best performance)
5. **Support Vector Machine** - Kernel-based classifier
6. **Naive Bayes** - Probabilistic classifier
7. **K-Nearest Neighbors** - Instance-based learning

---

## Docker Deployment

### Build and Run with Docker
```bash
# Build the image
docker build -t heart-disease-dashboard .

# Run the container
docker run -p 8501:8501 heart-disease-dashboard
```

### Using Docker Compose (Recommended)
```bash
# Start the dashboard
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the dashboard
docker-compose down
```

### Access the Dashboard

Open your browser to: **http://localhost:8501**

### Docker Image Size

- Base image: ~500 MB
- Final image: ~800 MB (with dependencies)

---

## Modifications from Original Lab

### Original Streamlit Lab

The original lab typically includes:
- Single-page application
- Basic form inputs
- Simple prediction display
- Minimal styling

### My Enhancements

#### 1. **Multi-Page Architecture** 
- 5 comprehensive pages vs 1 page
- Organized navigation
- Modular code structure

#### 2. **Advanced Visualizations** 
- Interactive Plotly charts
- Real-time data updates
- Multiple chart types (bar, line, scatter, heatmap, etc.)
- Customizable visualizations

#### 3. **MLflow Integration** 
- Complete experiment tracking
- Run comparison
- Parameter analysis
- Artifact management

#### 4. **Model Comparison System** 
- 7 models instead of 1
- Side-by-side comparison
- ROC curves and confusion matrices
- Performance ranking

#### 5. **Professional UI/UX** 
- Custom CSS styling
- Responsive design
- Loading animations
- Color-coded results

#### 6. **Data Analysis Module** 
- Comprehensive EDA
- Correlation analysis
- Feature distributions
- Data quality reports

#### 7. **Docker Deployment**
- Containerized application
- Docker Compose configuration
- Easy deployment
- Production-ready

#### 8. **Modular Code Structure** 
- Utility functions
- Reusable components
- Clean separation of concerns
- Well-documented code

### Comparison Table

| Feature | Original Lab | My Implementation |
|---------|--------------|-------------------|
| **Pages** | 1 | 5 |
| **Models** | 1 | 7 |
| **Visualizations** | Basic | Advanced (Plotly) |
| **MLflow** | Not included | Fully integrated |
| **Model Comparison** | None | Comprehensive |
| **Data Analysis** | None | Full EDA suite |
| **Docker** | No | Yes |
| **Code Lines** | ~100 | ~2000+ |

---



