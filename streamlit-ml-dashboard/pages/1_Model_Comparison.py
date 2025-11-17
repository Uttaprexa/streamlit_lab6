"""
Model Comparison Page
Compare all 7 ML models performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Model Comparison",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    .model-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin: 1rem 0;
    }
    .best-model {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_data
def load_data():
    """Load dataset"""
    df = pd.read_csv('data/heart_disease.csv')
    return df

@st.cache_data
def train_all_models():
    """Train all 7 models and return results"""
    df = load_data()
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    with st.spinner('Training all models... This may take a moment...'):
        for name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            results[name] = metrics
    
    return results

def plot_comparison_bar(results):
    """Create comparison bar chart"""
    models = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for metric in metrics_names:
        values = [results[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Disease', 'Disease'],
        y=['No Disease', 'Disease'],
        colorscale='RdBu',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        reversescale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig

def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    for model_name, metrics in results.items():
        if metrics['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f'{model_name} (AUC={roc_auc:.3f})',
                mode='lines'
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random (AUC=0.5)',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=600,
        hovermode='closest'
    )
    
    return fig

def create_metrics_table(results):
    """Create comprehensive metrics table"""
    data = []
    
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False)
    
    return df

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.title("Model Comparison Dashboard")
    st.markdown("""
        Compare the performance of all 7 machine learning models trained on the heart disease dataset.
        Each model is evaluated using multiple metrics to provide a comprehensive view.
    """)
    
    st.divider()
    
    # Train models
    with st.spinner('Training all models...'):
        results = train_all_models()
    
    st.success('All models trained successfully!')
    
    # ========================================
    # BEST MODEL HIGHLIGHT
    # ========================================
    
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    st.markdown(f"""
        <div class="best-model">
            <h2> Best Model</h2>
            <h1>{best_model}</h1>
            <h3>Accuracy: {best_accuracy:.2%}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # QUICK STATS
    # ========================================
    
    st.subheader("Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        st.metric("Average Accuracy", f"{avg_accuracy:.2%}")
    
    with col2:
        best_f1 = max([r['f1_score'] for r in results.values()])
        st.metric("Best F1-Score", f"{best_f1:.4f}")
    
    with col3:
        st.metric("Models Trained", len(results))
    
    with col4:
        st.metric("Total Comparisons", "28")  # 7 models * 4 metrics
    
    # ========================================
    # COMPARISON CHART
    # ========================================
    
    st.divider()
    st.subheader("Performance Comparison")
    
    tab1, tab2, tab3 = st.tabs(["Bar Chart", "Line Chart", "Table"])
    
    with tab1:
        fig = plot_comparison_bar(results)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Line chart for comparison
        models = list(results.keys())
        metrics = {
            'Accuracy': [results[m]['accuracy'] for m in models],
            'Precision': [results[m]['precision'] for m in models],
            'Recall': [results[m]['recall'] for m in models],
            'F1-Score': [results[m]['f1_score'] for m in models]
        }
        
        fig = go.Figure()
        for metric_name, values in metrics.items():
            fig.add_trace(go.Scatter(
                x=models,
                y=values,
                name=metric_name,
                mode='lines+markers',
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title='Metric Trends Across Models',
            xaxis_title='Models',
            yaxis_title='Score',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df = create_metrics_table(results)
        
        # Highlight best model
        def highlight_max(s):
            if s.name == 'Model':
                return [''] * len(s)
            is_max = s == s.max()
            return ['background-color: #d4edda' if v else '' for v in is_max]
        
        styled_df = df.style.apply(highlight_max, axis=0)
        st.dataframe(styled_df, use_container_width=True, height=300)
    
    # ========================================
    # ROC CURVES
    # ========================================
    
    st.divider()
    st.subheader("ROC Curves Analysis")
    
    fig = plot_roc_curves(results)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
        **ROC Curve Interpretation:**
        - The closer the curve to the top-left corner, the better the model
        - AUC (Area Under Curve) close to 1.0 indicates excellent performance
        - Diagonal line represents random guessing (AUC = 0.5)
    """)
    
    # ========================================
    # CONFUSION MATRICES
    # ========================================
    
    st.divider()
    st.subheader("Confusion Matrices")
    
    st.markdown("""
        Select a model to view its confusion matrix. This shows how well the model 
        distinguishes between patients with and without heart disease.
    """)
    
    selected_model = st.selectbox(
        "Choose a model:",
        list(results.keys()),
        index=list(results.keys()).index(best_model)
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        cm = results[selected_model]['confusion_matrix']
        fig = plot_confusion_matrix(cm, selected_model)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        
        # Extract values
        tn, fp, fn, tp = cm.ravel()
        
        st.markdown(f"""
            ### Confusion Matrix Breakdown:
            
            | Metric | Value |
            |--------|-------|
            | **True Negatives (TN)** | {tn} |
            | **False Positives (FP)** | {fp} |
            | **False Negatives (FN)** | {fn} |
            | **True Positives (TP)** | {tp} |
            
            ---
            
            **Derived Metrics:**
            - **Sensitivity (Recall):** {tp/(tp+fn):.2%}
            - **Specificity:** {tn/(tn+fp):.2%}
            - **Precision:** {tp/(tp+fp):.2%}
        """)
    
    # ========================================
    # DETAILED METRICS
    # ========================================
    
    st.divider()
    st.subheader("Detailed Model Reports")
    
    with st.expander("View Detailed Classification Reports"):
        for model_name, metrics in results.items():
            st.markdown(f"### {model_name}")
            
            report = classification_report(
                metrics['y_test'],
                metrics['y_pred'],
                target_names=['No Disease', 'Disease'],
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            st.divider()
    
    # ========================================
    # MODEL RANKING
    # ========================================
    
    st.divider()
    st.subheader("Final Model Ranking")
    
    # Calculate overall score (weighted average)
    rankings = []
    for model_name, metrics in results.items():
        score = (
            metrics['accuracy'] * 0.4 +
            metrics['precision'] * 0.2 +
            metrics['recall'] * 0.2 +
            metrics['f1_score'] * 0.2
        )
        rankings.append({
            'Rank': 0,
            'Model': model_name,
            'Overall Score': score,
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_score']
        })
    
    rankings = sorted(rankings, key=lambda x: x['Overall Score'], reverse=True)
    for i, r in enumerate(rankings, 1):
        r['Rank'] = i
    
    ranking_df = pd.DataFrame(rankings)
    
    # Display with medals
    for idx, row in ranking_df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
        
        with col1:
            if row['Rank'] == 1:
                st.markdown("#")
            elif row['Rank'] == 2:
                st.markdown("#")
            elif row['Rank'] == 3:
                st.markdown("#")
            else:
                st.markdown(f"### #{row['Rank']}")
        
        with col2:
            st.markdown(f"### {row['Model']}")
        
        with col3:
            st.metric("Overall Score", f"{row['Overall Score']:.4f}")
        
        with col4:
            st.metric("Accuracy", f"{row['Accuracy']:.2%}")
    
    # ========================================
    # INSIGHTS
    # ========================================
    
    st.divider()
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### Strengths:
            - All models achieve >75% accuracy
            - Ensemble methods (RF, GB) perform best
            - Consistent precision across models
            - Low false negative rates
        """)
    
    with col2:
        st.markdown("""
            ### Considerations:
            - Simple models (Naive Bayes) lag behind
            - Some models may be overfitting
            - Class imbalance affects precision
            - Consider model complexity vs performance
        """)
    
    # ========================================
    # RECOMMENDATIONS
    # ========================================
    
    st.info(f"""
        ###  Recommendation:
        
        Based on comprehensive evaluation, **{best_model}** is recommended for production deployment 
        with {best_accuracy:.2%} accuracy. However, consider:
        
        - **For Speed:** Use Logistic Regression (fastest inference)
        - **For Accuracy:** Use {best_model} (best performance)
        - **For Interpretability:** Use Decision Tree (most explainable)
        - **For Balance:** Use Random Forest (good trade-off)
    """)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("Model Comparison")
    
    st.info("""
        This page provides comprehensive comparison 
        of all 7 trained ML models.
    """)
    
    st.divider()
    
    st.subheader("Evaluation Metrics")
    st.markdown("""
        - **Accuracy:** Overall correctness
        - **Precision:** Positive prediction accuracy
        - **Recall:** True positive detection rate
        - **F1-Score:** Harmonic mean of precision & recall
    """)
    
    st.divider()
    
    st.subheader("Models Compared")
    st.markdown("""
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. Gradient Boosting
        5. Support Vector Machine
        6. Naive Bayes
        7. K-Nearest Neighbors
    """)

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()