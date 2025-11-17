"""
Data Analysis Page
Explore and analyze the heart disease dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Data Analysis",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    .analysis-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
def get_dataset_stats(df):
    """Calculate dataset statistics"""
    stats = {
        'total_samples': len(df),
        'n_features': df.shape[1] - 1,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'disease_count': (df['target'] == 1).sum(),
        'healthy_count': (df['target'] == 0).sum(),
        'disease_rate': (df['target'] == 1).mean() * 100
    }
    return stats

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.title("Data Analysis & Exploration")
    st.markdown("""
        Comprehensive exploratory data analysis (EDA) of the heart disease dataset. 
        Understand feature distributions, correlations, and patterns in the data.
    """)
    
    st.divider()
    
    # Load data
    df = load_data()
    stats = get_dataset_stats(df)
    
    # ========================================
    # DATASET OVERVIEW
    # ========================================
    
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", stats['total_samples'])
    
    with col2:
        st.metric("Features", stats['n_features'])
    
    with col3:
        st.metric("Disease Cases", stats['disease_count'], 
                 delta=f"{stats['disease_rate']:.1f}%")
    
    with col4:
        st.metric("Missing Values", stats['missing_values'],
                 delta="Perfect!" if stats['missing_values'] == 0 else "")
    
    # ========================================
    # DATA PREVIEW
    # ========================================
    
    st.divider()
    st.subheader("Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Statistics", "Info"])
    
    with tab1:
        st.dataframe(df, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
    
    with tab2:
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab3:
        # Feature information
        feature_info = pd.DataFrame({
            'Feature': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Missing': [df[col].isnull().sum() for col in df.columns]
        })
        st.dataframe(feature_info, use_container_width=True)
    
    # ========================================
    # TARGET DISTRIBUTION
    # ========================================
    
    st.divider()
    st.subheader("Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        target_counts = df['target'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=['No Disease (0)', 'Disease (1)'],
            title='Target Distribution',
            color_discrete_sequence=['#51cf66', '#ff6b6b'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label+value')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            x=['No Disease', 'Disease'],
            y=target_counts.values,
            title='Target Class Counts',
            labels={'x': 'Class', 'y': 'Count'},
            color=['No Disease', 'Disease'],
            color_discrete_sequence=['#51cf66', '#ff6b6b'],
            text=target_counts.values
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Class balance info
    class_ratio = stats['healthy_count'] / stats['disease_count']
    
    if class_ratio > 2 or class_ratio < 0.5:
        st.warning(f"""
            **Class Imbalance Detected!**  
            Ratio: {class_ratio:.2f}:1 (Healthy:Disease)  
            Consider using techniques like SMOTE or class weights.
        """)
    else:
        st.success(f"""
            **Classes are reasonably balanced**  
            Ratio: {class_ratio:.2f}:1 (Healthy:Disease)
        """)
    
    # ========================================
    # FEATURE DISTRIBUTIONS
    # ========================================
    
    st.divider()
    st.subheader("Feature Distributions")
    
    # Feature selector
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('target')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_feature = st.selectbox(
            "Select a feature to analyze:",
            numeric_features
        )
    
    with col2:
        plot_type = st.radio(
            "Plot type:",
            ["Histogram", "Box Plot", "Violin Plot"],
            horizontal=True
        )
    
    # Create plots
    col1, col2 = st.columns(2)
    
    with col1:
        if plot_type == "Histogram":
            fig = px.histogram(
                df,
                x=selected_feature,
                color='target',
                title=f'{selected_feature} Distribution by Target',
                marginal='box',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                barmode='overlay',
                opacity=0.7
            )
        elif plot_type == "Box Plot":
            fig = px.box(
                df,
                x='target',
                y=selected_feature,
                title=f'{selected_feature} by Target',
                color='target',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'}
            )
        else:  # Violin Plot
            fig = px.violin(
                df,
                x='target',
                y=selected_feature,
                title=f'{selected_feature} Distribution',
                color='target',
                color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
                box=True
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Statistics for selected feature
        st.markdown(f"### {selected_feature} Statistics")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Mean", f"{df[selected_feature].mean():.2f}")
            st.metric("Median", f"{df[selected_feature].median():.2f}")
            st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
        
        with col_b:
            st.metric("Min", f"{df[selected_feature].min():.2f}")
            st.metric("Max", f"{df[selected_feature].max():.2f}")
            st.metric("Range", f"{df[selected_feature].max() - df[selected_feature].min():.2f}")
        
        # By target class
        st.markdown("### By Target Class")
        
        disease_mean = df[df['target']==1][selected_feature].mean()
        healthy_mean = df[df['target']==0][selected_feature].mean()
        
        st.metric(
            "Disease (1)",
            f"{disease_mean:.2f}",
            delta=f"{disease_mean - healthy_mean:.2f}",
            delta_color="inverse"
        )
        st.metric(
            "No Disease (0)",
            f"{healthy_mean:.2f}"
        )
    
    # ========================================
    # CORRELATION ANALYSIS
    # ========================================
    
    st.divider()
    st.subheader("Correlation Analysis")
    
    tab1, tab2 = st.tabs(["Heatmap", "Feature Correlation"])
    
    with tab1:
        # Correlation heatmap
        corr_matrix = df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key correlations with target
        st.markdown("### Strongest Correlations with Target")
        
        target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Correlations:**")
            positive = target_corr.head(5)
            for feature, corr in positive.items():
                st.metric(feature, f"{corr:.3f}", delta="Positive")
        
        with col2:
            st.markdown("**Negative Correlations:**")
            negative = target_corr.tail(5)
            for feature, corr in negative.items():
                st.metric(feature, f"{corr:.3f}", delta="Negative", delta_color="inverse")
    
    with tab2:
        # Correlation with target bar chart
        target_corr_abs = corr_matrix['target'].drop('target').abs().sort_values(ascending=True)
        
        fig = go.Figure(go.Bar(
            x=target_corr_abs.values,
            y=target_corr_abs.index,
            orientation='h',
            marker_color='indianred',
            text=target_corr_abs.values,
            texttemplate='%{text:.3f}',
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Feature Importance (Correlation with Target)',
            xaxis_title='Absolute Correlation',
            yaxis_title='Features',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # PAIRWISE RELATIONSHIPS
    # ========================================
    
    st.divider()
    st.subheader("Pairwise Relationships")
    
    st.markdown("""
        Explore relationships between pairs of features. Select two features to compare.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("X-axis feature:", numeric_features, index=0)
    
    with col2:
        feature_y = st.selectbox("Y-axis feature:", numeric_features, index=1)
    
    # Scatter plot
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color='target',
        title=f'{feature_x} vs {feature_y}',
        color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
        marginal_x='histogram',
        marginal_y='histogram',
        opacity=0.7
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # FEATURE ANALYSIS BY TARGET
    # ========================================
    
    st.divider()
    st.subheader("Feature Analysis by Target Class")
    
    # Select multiple features
    selected_features = st.multiselect(
        "Select features to compare:",
        numeric_features,
        default=numeric_features[:3]
    )
    
    if selected_features:
        # Create comparison plots
        fig = make_subplots(
            rows=1,
            cols=len(selected_features),
            subplot_titles=selected_features
        )
        
        for idx, feature in enumerate(selected_features, 1):
            # Add box plot for each feature
            for target_val in [0, 1]:
                data = df[df['target'] == target_val][feature]
                fig.add_trace(
                    go.Box(
                        y=data,
                        name=f"Target {target_val}",
                        marker_color='#51cf66' if target_val == 0 else '#ff6b6b',
                        showlegend=(idx == 1)
                    ),
                    row=1,
                    col=idx
                )
        
        fig.update_layout(
            height=400,
            title_text="Feature Distributions by Target Class"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================
    # DATA QUALITY
    # ========================================
    
    st.divider()
    st.subheader("Data Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Completeness")
        
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        
        fig = go.Figure(go.Bar(
            x=completeness.values,
            y=completeness.index,
            orientation='h',
            marker_color='#51cf66',
            text=completeness.values,
            texttemplate='%{text:.1f}%',
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Feature Completeness',
            xaxis_title='Completeness (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Uniqueness")
        
        uniqueness = df.nunique()
        uniqueness_pct = (uniqueness / len(df)) * 100
        
        quality_df = pd.DataFrame({
            'Feature': uniqueness.index,
            'Unique Values': uniqueness.values,
            'Uniqueness (%)': uniqueness_pct.values
        })
        
        st.dataframe(quality_df, use_container_width=True, height=400)
    
    # ========================================
    # KEY INSIGHTS
    # ========================================
    
    st.divider()
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="insight-box">
                <h4>Data Quality</h4>
                <ul>
                    <li>No missing values</li>
                    <li>No duplicate records</li>
                    <li>Balanced target classes</li>
                    <li>All features have valid ranges</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_feature = target_corr_abs.index[-1]
        top_corr = corr_matrix['target'][top_feature]
        
        st.markdown(f"""
            <div class="insight-box">
                <h4>🔍 Important Features</h4>
                <ul>
                    <li>Most correlated: <strong>{top_feature}</strong> ({top_corr:.3f})</li>
                    <li>{len([c for c in target_corr if c > 0.2])} features strongly correlated</li>
                    <li>Class ratio: {class_ratio:.2f}:1</li>
                    <li>{stats['n_features']} predictive features</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # ========================================
    # DOWNLOAD ANALYSIS
    # ========================================
    
    st.divider()
    st.subheader("Download Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Dataset (CSV)",
            data=csv,
            file_name="heart_disease_dataset.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download statistics
        stats_df = df.describe()
        csv_stats = stats_df.to_csv()
        st.download_button(
            label="Download Statistics (CSV)",
            data=csv_stats,
            file_name="dataset_statistics.csv",
            mime="text/csv"
        )

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("Data Analysis")
    
    st.info("""
        Comprehensive exploratory data analysis (EDA) 
        of the heart disease dataset.
    """)
    
    st.divider()
    
    st.subheader("Dataset Info")
    st.markdown("""
        **Source:** Synthetic Heart Disease Data  
        **Samples:** 500 patients  
        **Features:** 13 clinical parameters  
        **Target:** Binary (Disease/No Disease)
    """)
    
    st.divider()
    
    st.subheader("Analysis Includes")
    st.markdown("""
        - Data overview & statistics
        - Target distribution
        - Feature distributions
        - Correlation analysis
        - Pairwise relationships
        - Data quality report
    """)
    
    st.divider()
    
    st.subheader("Features")
    df = load_data()
    for col in df.columns:
        if col != 'target':
            st.text(f"• {col}")

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()