"""
MLflow Explorer Page
View and analyze MLflow experiment tracking data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="MLflow Explorer",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    .experiment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .run-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin: 0.5rem 0;
    }
    .metric-badge {
        display: inline-block;
        background-color: #e7f5ff;
        color: #1971c2;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_data
def check_mlflow_available():
    """Check if MLflow tracking data exists"""
    mlruns_path = Path('mlruns')
    return mlruns_path.exists() and any(mlruns_path.iterdir())

@st.cache_data
def get_all_experiments():
    """Get all MLflow experiments"""
    if not check_mlflow_available():
        return None
    
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        client = MlflowClient()
        experiments = client.search_experiments()
        return experiments
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return None

@st.cache_data
def get_runs_from_experiment(experiment_id):
    """Get all runs from an experiment"""
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"]
        )
        return runs
    except Exception as e:
        st.error(f"Error loading runs: {e}")
        return None

def create_sample_mlflow_data():
    """Create sample MLflow data for demonstration"""
    # Sample data to show what MLflow tracking looks like
    sample_runs = pd.DataFrame({
        'run_name': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 
                     'SVM', 'Decision Tree', 'Naive Bayes', 'KNN'],
        'accuracy': [0.8500, 0.9667, 0.9500, 0.8500, 0.8000, 0.7800, 0.8300],
        'precision': [0.8528, 0.9682, 0.9520, 0.8467, 0.8025, 0.7810, 0.8315],
        'recall': [0.8500, 0.9667, 0.9500, 0.8500, 0.8000, 0.7800, 0.8300],
        'f1_score': [0.8511, 0.9661, 0.9510, 0.8458, 0.8012, 0.7805, 0.8308],
        'training_time': [0.04, 0.16, 0.15, 0.02, 0.02, 0.01, 0.01],
        'timestamp': pd.date_range('2025-11-01', periods=7, freq='D')
    })
    return sample_runs

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.title("MLflow Experiment Explorer")
    st.markdown("""
        View and analyze your MLflow experiment tracking data. This page shows all 
        training runs, hyperparameters, metrics, and artifacts logged during model development.
    """)
    
    st.divider()
    
    # Check if MLflow data exists
    mlflow_available = check_mlflow_available()
    
    if not mlflow_available:
        st.warning("""
            **No MLflow tracking data found!**
            
            MLflow tracking data is not available in this project. This could mean:
            1. Models haven't been trained with MLflow tracking yet
            2. The `mlruns` directory doesn't exist
            3. MLflow wasn't configured during training
            
            **Showing sample/demo data instead...**
        """)
        
        # Show sample data
        st.info("""
            **How to generate real MLflow data:**
            
            1. Run your model training with MLflow tracking enabled
            2. Ensure MLflow logs to `./mlruns` directory
            3. Refresh this page to see your actual experiment data
            
            For now, explore the sample data below to understand MLflow capabilities!
        """)
        
        sample_runs = create_sample_mlflow_data()
        show_sample_dashboard(sample_runs)
        
    else:
        # Show real MLflow data
        show_mlflow_dashboard()

def show_sample_dashboard(sample_runs):
    """Show dashboard with sample data"""
    
    # ========================================
    # OVERVIEW STATS
    # ========================================
    
    st.subheader("Experiment Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(sample_runs))
    
    with col2:
        best_accuracy = sample_runs['accuracy'].max()
        st.metric("Best Accuracy", f"{best_accuracy:.2%}")
    
    with col3:
        avg_training_time = sample_runs['training_time'].mean()
        st.metric("Avg Training Time", f"{avg_training_time:.2f}s")
    
    with col4:
        total_experiments = 1
        st.metric("Total Experiments", total_experiments)
    
    # ========================================
    # RUNS TABLE
    # ========================================
    
    st.divider()
    st.subheader("All Training Runs")
    
    # Style the dataframe
    styled_df = sample_runs[['run_name', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time']].copy()
    styled_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)']
    
    # Highlight best model
    def highlight_max(s):
        if s.name in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            is_max = s == s.max()
            return ['background-color: #d4edda' if v else '' for v in is_max]
        return [''] * len(s)
    
    styled_table = styled_df.style.apply(highlight_max, axis=0).format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'Training Time (s)': '{:.2f}'
    })
    
    st.dataframe(styled_table, use_container_width=True, height=300)
    
    # ========================================
    # METRICS COMPARISON
    # ========================================
    
    st.divider()
    st.subheader("Metrics Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Line Chart", "Scatter Plot"])
    
    with tab1:
        # Bar chart comparison
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=sample_runs['run_name'],
                y=sample_runs[metric],
                text=sample_runs[metric].apply(lambda x: f'{x:.3f}'),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Line chart over time
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Scatter(
                x=sample_runs['timestamp'],
                y=sample_runs[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers',
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title='Metrics Over Time',
            xaxis_title='Training Date',
            yaxis_title='Score',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Scatter plot: Accuracy vs Training Time
        fig = px.scatter(
            sample_runs,
            x='training_time',
            y='accuracy',
            size='f1_score',
            color='run_name',
            hover_data=['precision', 'recall'],
            title='Accuracy vs Training Time (size = F1-Score)',
            labels={
                'training_time': 'Training Time (seconds)',
                'accuracy': 'Accuracy',
                'run_name': 'Model'
            }
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # BEST MODEL DETAILS
    # ========================================
    
    st.divider()
    st.subheader("Best Model Details")
    
    best_idx = sample_runs['accuracy'].idxmax()
    best_model = sample_runs.iloc[best_idx]
    
    st.markdown(f"""
        <div class="experiment-card">
            <h2>{best_model['run_name']}</h2>
            <h3>Accuracy: {best_model['accuracy']:.2%}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### Performance Metrics
        """)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                f"{best_model['accuracy']:.4f}",
                f"{best_model['precision']:.4f}",
                f"{best_model['recall']:.4f}",
                f"{best_model['f1_score']:.4f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("""
            ### Training Info
        """)
        
        info_df = pd.DataFrame({
            'Property': ['Training Time', 'Timestamp', 'Status'],
            'Value': [
                f"{best_model['training_time']:.2f} seconds",
                best_model['timestamp'].strftime('%Y-%m-%d'),
                'Completed'
            ]
        })
        
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # ========================================
    # HYPERPARAMETERS (Sample)
    # ========================================
    
    st.divider()
    st.subheader("Hyperparameters")
    
    with st.expander("View Sample Hyperparameters"):
        # Sample hyperparameters
        if 'Random Forest' in best_model['run_name']:
            params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        elif 'Gradient Boosting' in best_model['run_name']:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.9,
                'random_state': 42
            }
        else:
            params = {
                'max_iter': 1000,
                'random_state': 42
            }
        
        params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    # ========================================
    # EXPERIMENT TIMELINE
    # ========================================
    
    st.divider()
    st.subheader("Experiment Timeline")
    
    # Create Gantt-like timeline
    fig = px.timeline(
        sample_runs,
        x_start='timestamp',
        x_end=sample_runs['timestamp'] + pd.Timedelta(hours=1),
        y='run_name',
        color='accuracy',
        title='Training Timeline',
        labels={'run_name': 'Model', 'timestamp': 'Date'},
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # DOWNLOAD RESULTS
    # ========================================
    
    st.divider()
    st.subheader("Export Results")
    
    csv = sample_runs.to_csv(index=False)
    st.download_button(
        label="Download All Results as CSV",
        data=csv,
        file_name="mlflow_experiment_results.csv",
        mime="text/csv"
    )

def show_mlflow_dashboard():
    """Show dashboard with real MLflow data"""
    
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Get experiments
    experiments = get_all_experiments()
    
    if not experiments or len(experiments) == 0:
        st.warning("No experiments found in MLflow tracking.")
        return
    
    # ========================================
    # EXPERIMENT SELECTOR
    # ========================================
    
    st.subheader("Select Experiment")
    
    experiment_names = [exp.name for exp in experiments]
    selected_exp_name = st.selectbox(
        "Choose an experiment:",
        experiment_names
    )
    
    # Get selected experiment
    selected_exp = next(exp for exp in experiments if exp.name == selected_exp_name)
    
    st.info(f"""
        **Experiment ID:** `{selected_exp.experiment_id}`  
        **Artifact Location:** `{selected_exp.artifact_location}`
    """)
    
    # ========================================
    # LOAD RUNS
    # ========================================
    
    runs = get_runs_from_experiment(selected_exp.experiment_id)
    
    if runs is None or len(runs) == 0:
        st.warning(f"No runs found in experiment: {selected_exp_name}")
        return
    
    st.success(f"Found {len(runs)} runs in this experiment!")
    
    # ========================================
    # OVERVIEW STATS
    # ========================================
    
    st.divider()
    st.subheader("Experiment Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(runs))
    
    with col2:
        if 'metrics.accuracy' in runs.columns:
            best_accuracy = runs['metrics.accuracy'].max()
            st.metric("Best Accuracy", f"{best_accuracy:.2%}")
        else:
            st.metric("Best Accuracy", "N/A")
    
    with col3:
        if 'metrics.training_time_seconds' in runs.columns:
            avg_time = runs['metrics.training_time_seconds'].mean()
            st.metric("Avg Training Time", f"{avg_time:.2f}s")
        else:
            st.metric("Avg Training Time", "N/A")
    
    with col4:
        st.metric("Experiments", len(experiments))
    
    # ========================================
    # RUNS TABLE
    # ========================================
    
    st.divider()
    st.subheader("All Training Runs")
    
    # Select relevant columns
    display_cols = ['tags.mlflow.runName', 'metrics.accuracy', 'metrics.precision', 
                    'metrics.recall', 'metrics.f1_score', 'start_time']
    
    available_cols = [col for col in display_cols if col in runs.columns]
    
    if available_cols:
        display_df = runs[available_cols].copy()
        display_df.columns = [col.replace('metrics.', '').replace('tags.mlflow.', '').replace('_', ' ').title() 
                               for col in display_df.columns]
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(runs.head(), use_container_width=True)
    
    # ========================================
    # METRICS VISUALIZATION
    # ========================================
    
    st.divider()
    st.subheader("Metrics Visualization")
    
    if 'metrics.accuracy' in runs.columns:
        # Create comparison chart
        fig = go.Figure()
        
        metric_cols = [col for col in runs.columns if col.startswith('metrics.') and col != 'metrics.training_time_seconds']
        
        for col in metric_cols:
            metric_name = col.replace('metrics.', '').replace('_', ' ').title()
            fig.add_trace(go.Bar(
                name=metric_name,
                x=runs['tags.mlflow.runName'] if 'tags.mlflow.runName' in runs.columns else range(len(runs)),
                y=runs[col],
                text=runs[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A'),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics data available to visualize")
    
    # ========================================
    # BEST RUN DETAILS
    # ========================================
    
    st.divider()
    st.subheader("Best Run")
    
    if 'metrics.accuracy' in runs.columns:
        best_run_idx = runs['metrics.accuracy'].idxmax()
        best_run = runs.iloc[best_run_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Metrics")
            for col in runs.columns:
                if col.startswith('metrics.'):
                    metric_name = col.replace('metrics.', '').replace('_', ' ').title()
                    value = best_run[col]
                    if pd.notna(value):
                        st.metric(metric_name, f"{value:.4f}")
        
        with col2:
            st.markdown("### Parameters")
            for col in runs.columns:
                if col.startswith('params.'):
                    param_name = col.replace('params.', '').replace('_', ' ').title()
                    value = best_run[col]
                    if pd.notna(value):
                        st.text(f"{param_name}: {value}")
    
    # ========================================
    # DOWNLOAD
    # ========================================
    
    st.divider()
    st.subheader("Export Data")
    
    csv = runs.to_csv(index=False)
    st.download_button(
        label="Download Runs Data",
        data=csv,
        file_name=f"mlflow_{selected_exp_name}_runs.csv",
        mime="text/csv"
    )

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("MLflow Explorer")
    
    st.info("""
        Explore your MLflow experiment tracking data, 
        compare runs, and analyze model performance.
    """)
    
    st.divider()
    
    st.subheader("What is MLflow?")
    st.markdown("""
        **MLflow** is an open-source platform for managing 
        the ML lifecycle, including:
        
        - **Tracking:** Log parameters, metrics, artifacts
        - **Projects:** Package code for reproducibility
        - **Models:** Deploy models to various platforms
        - **Registry:** Store and version models
    """)
    
    st.divider()
    
    st.subheader("Features")
    st.markdown("""
        - View all experiments
        - Compare run metrics
        - Analyze hyperparameters
        - Track training timeline
        - Export results
    """)
    
    st.divider()
    
    st.markdown("""
        **Tip:** To generate MLflow data, 
        train your models with MLflow tracking enabled!
    """)

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()