import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ============================================
# PAGE CONFIG 
# ============================================

st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subheader styling */
    .sub-header {
        font-size: 1.5rem;
        color: #262730;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .info-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin: 1rem 0;
    }
    
    /* Feature box */
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Metric card */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_data
def load_data():
    """Load heart disease dataset"""
    try:
        df = pd.read_csv('data/heart_disease.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please add 'data/heart_disease.csv'")
        return None

@st.cache_data
def get_stats():
    """Get dataset statistics"""
    df = load_data()
    if df is not None:
        return {
            'total_samples': len(df),
            'n_features': df.shape[1] - 1,
            'disease_cases': (df['target'] == 1).sum(),
            'healthy_cases': (df['target'] == 0).sum(),
            'disease_rate': (df['target'] == 1).mean() * 100
        }
    return {}

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header with custom styling
    st.markdown(
        '<h1 class="main-header">Heart Disease Prediction System</h1>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p class="sub-header">AI-Powered Medical Diagnosis Dashboard using 7 ML Models</p>',
        unsafe_allow_html=True
    )
    
    # ========================================
    # HERO SECTION
    # ========================================
    
    st.write("")  # Spacing
    
    # Quick info boxes
    col1, col2, col3, col4 = st.columns(4)
    
    stats = get_stats()
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #FF4B4B;"></h3>
                <h2>7</h2>
                <p>ML Models</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #4CAF50;"></h3>
                <h2>95%</h2>
                <p>Best Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2196F3;"></h3>
                <h2>{stats.get('total_samples', 0)}</h2>
                <p>Total Samples</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #FF9800;"></h3>
                <h2>Real-time</h2>
                <p>Predictions</p>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================
    # INTRODUCTION
    # ========================================
    
    st.write("---")
    
    st.markdown("""
        ### Welcome to the Heart Disease Prediction Dashboard!
        
        This interactive application uses **machine learning** to predict the risk of heart disease 
        based on clinical parameters. Built with Streamlit and powered by MLflow experiment tracking.
    """)
    
    # Features in columns
    st.write("")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            #### Key Features:
            - **Explore** comprehensive heart disease dataset
            - **Compare** 7 different ML algorithms
            - **Predict** heart disease risk in real-time
            - **Track** experiments with MLflow integration
            - **Analyze** data with interactive visualizations
        """)
    
    with col2:
        st.markdown("""
            #### ML Models Included:
            1. Logistic Regression
            2. Decision Tree
            3. Random Forest 
            4. Gradient Boosting (Best: 95%)
            5. Support Vector Machine (SVM)
            6. Naive Bayes
            7. K-Nearest Neighbors (KNN)
        """)
    
    # ========================================
    # QUICK STATS
    # ========================================
    
    st.write("---")
    st.subheader("Dataset Overview")
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Samples",
                stats['total_samples'],
                help="Total number of patient records"
            )
        
        with col2:
            st.metric(
                "Features",
                stats['n_features'],
                help="Number of clinical parameters"
            )
        
        with col3:
            st.metric(
                "Disease Cases",
                stats['disease_cases'],
                delta=f"{stats['disease_rate']:.1f}%",
                delta_color="inverse",
                help="Patients with heart disease"
            )
        
        with col4:
            st.metric(
                "Healthy Cases",
                stats['healthy_cases'],
                delta=f"{100-stats['disease_rate']:.1f}%",
                help="Patients without heart disease"
            )
    
    # ========================================
    # VISUALIZATION
    # ========================================
    
    st.write("")
    df = load_data()
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Target Distribution")
            
            # Create pie chart
            target_counts = df['target'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['No Disease', 'Disease'],
                title='Distribution of Heart Disease Cases',
                color_discrete_sequence=['#4CAF50', '#FF4B4B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            
            # Create histogram
            fig = px.histogram(
                df,
                x='age',
                color='target',
                title='Age Distribution by Target',
                labels={'target': 'Heart Disease'},
                color_discrete_map={0: '#4CAF50', 1: '#FF4B4B'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # HOW TO USE
    # ========================================
    
    st.write("---")
    
    with st.expander("How to Use This Dashboard", expanded=False):
        st.markdown("""
            ### Getting Started:
            
            1. **Model Comparison**
               - Navigate to compare all 7 ML models
               - View performance metrics side-by-side
               - Analyze confusion matrices and ROC curves
            
            2. **Live Prediction**
               - Enter patient clinical parameters
               - Select your preferred ML model
               - Get instant heart disease risk prediction
               - View probability scores
            
            3. **MLflow Explorer**
               - Browse all training experiments
               - Compare hyperparameters
               - Track model performance over time
            
            4. **Data Analysis**
               - Explore the heart disease dataset
               - View feature distributions
               - Analyze correlations
               - Interactive data filtering
            
            ### Navigation:
            - Use the **sidebar** on the left to switch between pages
            - Each page is focused on a specific functionality
            - All visualizations are interactive - click, zoom, and explore!
        """)
    
    # ========================================
    # QUICK START
    # ========================================
    
    st.write("---")
    st.subheader("Quick Start")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Compare Models", use_container_width=True):
            st.switch_page("pages/1_📊_Model_Comparison.py")
    
    with col2:
        if st.button("Make Prediction", use_container_width=True):
            st.switch_page("pages/2_🔮_Live_Prediction.py")
    
    with col3:
        if st.button("View Experiments", use_container_width=True):
            st.switch_page("pages/3_📈_MLflow_Explorer.py")
    
    with col4:
        if st.button("Analyze Data", use_container_width=True):
            st.switch_page("pages/4_📁_Data_Analysis.py")
    
    # ========================================
    # FOOTER
    # ========================================
    
    st.write("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p>Built with ❤️ using Streamlit | MLOps Lab Project</p>
            <p>Author: Uttapreksha Patel | Northeastern University</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

def sidebar():
    """Create sidebar"""
    st.sidebar.title("Navigation")
    st.sidebar.write("")
    
    st.sidebar.info("""
        **About This Dashboard:**
        
        This application demonstrates an end-to-end 
        MLOps workflow including:
        - Model training
        - Experiment tracking
        - Interactive deployment
        - Real-time predictions
    """)
    
    st.sidebar.write("---")
    
    st.sidebar.subheader("Project Stats")
    st.sidebar.metric("Total Models", "7")
    st.sidebar.metric("Best Accuracy", "95%")
    st.sidebar.metric("Dataset Size", "500")
    
    st.sidebar.write("---")
    
    st.sidebar.subheader("Links")
    st.sidebar.markdown("""
        - [GitHub Repository](#)
        - [Documentation](#)
        - [MLflow UI](#)
    """)
    
    st.sidebar.write("---")
    st.sidebar.caption("Version 1.0.0 | Last updated: Nov 2025")

# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    sidebar()
    main()