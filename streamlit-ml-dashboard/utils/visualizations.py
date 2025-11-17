"""
Visualization utility functions
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_confusion_matrix_plot(cm, title='Confusion Matrix'):
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Disease', 'Disease'],
        y=['No Disease', 'Disease'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        reversescale=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )
    
    return fig

def create_roc_curve(fpr, tpr, auc_score, model_name):
    """Create ROC curve plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        name=f'{model_name} (AUC={auc_score:.3f})',
        mode='lines',
        line=dict(width=3)
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
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    return fig

def create_feature_importance_plot(feature_names, importances):
    """Create feature importance bar plot"""
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker_color='indianred',
        text=df['Importance'],
        texttemplate='%{text:.3f}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500
    )
    
    return fig

def create_metrics_comparison_plot(results_dict):
    """Create metrics comparison bar plot"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [results_dict[model][metric] for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    return fig

def create_distribution_plot(df, column, target_column='target'):
    """Create distribution plot with target overlay"""
    fig = px.histogram(
        df,
        x=column,
        color=target_column,
        marginal='box',
        title=f'{column} Distribution by Target',
        color_discrete_map={0: '#51cf66', 1: '#ff6b6b'},
        barmode='overlay',
        opacity=0.7
    )
    
    return fig

def create_correlation_heatmap(corr_matrix):
    """Create correlation heatmap"""
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
    
    return fig