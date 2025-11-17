"""
Live Prediction Page
Make real-time heart disease predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Live Prediction",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
    <style>
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    .input-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f5ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #339af0;
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

@st.cache_resource
def train_models():
    """Train all models and return them with scaler"""
    df = load_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models, scaler

def create_gauge_chart(probability):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Disease Risk Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#51cf66'},
                {'range': [30, 70], 'color': '#ffd43b'},
                {'range': [70, 100], 'color': '#ff6b6b'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_risk_interpretation(probability):
    """Get risk interpretation based on probability"""
    if probability < 0.3:
        return "Low Risk", "low-risk", "The patient has a low risk of heart disease based on the provided parameters."
    elif probability < 0.7:
        return "Moderate Risk", "moderate-risk", "The patient has moderate risk. Further medical evaluation is recommended."
    else:
        return "High Risk", "high-risk", "The patient has high risk of heart disease. Immediate medical consultation is strongly advised."

# ============================================
# MAIN APP
# ============================================

def main():
    """Main application"""
    
    # Header
    st.title("Live Heart Disease Prediction")
    st.markdown("""
        Enter patient clinical parameters below to get an instant heart disease risk prediction 
        using our trained machine learning models.
    """)
    
    st.divider()
    
    # Load models
    with st.spinner('Loading models...'):
        models, scaler = train_models()
    
    # ========================================
    # MODEL SELECTION
    # ========================================
    
    st.subheader(" Select Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model_name = st.selectbox(
            "Choose a model for prediction:",
            list(models.keys()),
            index=3  # Default to Gradient Boosting
        )
    
    with col2:
        st.info(f"""
            **Selected Model:**  
            {selected_model_name}
        """)
    
    selected_model = models[selected_model_name]
    
    # ========================================
    # INPUT FEATURES
    # ========================================
    
    st.divider()
    st.subheader("Patient Information")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Basic Info", "Cardiac Metrics", "Test Results"])
    
    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider(
                "Age (years)",
                min_value=30,
                max_value=80,
                value=55,
                help="Patient's age in years"
            )
            
            sex = st.selectbox(
                "Sex",
                options=[0, 1],
                format_func=lambda x: "Female" if x == 0 else "Male",
                help="Patient's biological sex"
            )
            
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: [
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic"
                ][x],
                help="Type of chest pain experienced"
            )
        
        with col2:
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether fasting blood sugar is greater than 120 mg/dl"
            )
            
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: [
                    "Normal",
                    "ST-T Wave Abnormality",
                    "Left Ventricular Hypertrophy"
                ][x],
                help="Resting electrocardiographic results"
            )
            
            exang = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether exercise induces angina"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            trestbps = st.slider(
                "Resting Blood Pressure (mm Hg)",
                min_value=90,
                max_value=200,
                value=140,
                help="Resting blood pressure in mm Hg"
            )
            
            chol = st.slider(
                "Serum Cholesterol (mg/dl)",
                min_value=120,
                max_value=400,
                value=250,
                help="Serum cholesterol in mg/dl"
            )
            
            thalach = st.slider(
                "Maximum Heart Rate Achieved",
                min_value=70,
                max_value=200,
                value=150,
                help="Maximum heart rate achieved during exercise"
            )
        
        with col2:
            oldpeak = st.slider(
                "ST Depression Induced by Exercise",
                min_value=0.0,
                max_value=6.0,
                value=2.5,
                step=0.1,
                help="ST depression induced by exercise relative to rest"
            )
            
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[0, 1, 2],
                format_func=lambda x: [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ][x],
                help="Slope of the peak exercise ST segment"
            )
            
            ca = st.slider(
                "Number of Major Vessels (0-3)",
                min_value=0,
                max_value=3,
                value=1,
                help="Number of major vessels colored by fluoroscopy"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            thal = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                format_func=lambda x: [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect",
                    "Unknown"
                ][x],
                help="Thalassemia blood disorder status"
            )
        
        with col2:
            st.markdown("""
                <div class="info-box">
                <strong>ℹ️ About Thalassemia:</strong><br>
                A blood disorder that affects the body's 
                ability to produce hemoglobin and red blood cells.
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================
    # PREDICTION
    # ========================================
    
    st.divider()
    
    # Create feature array
    features = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "PREDICT RISK",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        with st.spinner('Analyzing patient data...'):
            # Make prediction
            prediction = selected_model.predict(features_scaled)[0]
            
            # Get probability if available
            if hasattr(selected_model, 'predict_proba'):
                probability = selected_model.predict_proba(features_scaled)[0][1]
            else:
                probability = float(prediction)
            
            # Get risk interpretation
            risk_level, risk_class, risk_message = get_risk_interpretation(probability)
            
            # Animation
            st.balloons() if prediction == 0 else st.snow()
            
            # ========================================
            # RESULTS
            # ========================================
            
            st.markdown("---")
            st.subheader(" Prediction Results")
            
            # Main prediction box
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box high-risk">
                        HIGH RISK OF HEART DISEASE
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box low-risk">
                        LOW RISK OF HEART DISEASE
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = create_gauge_chart(probability)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                
                st.markdown(f"""
                    ### Detailed Analysis:
                    
                    | Metric | Value |
                    |--------|-------|
                    | **Prediction** | {'Disease' if prediction == 1 else 'No Disease'} |
                    | **Probability** | {probability:.2%} |
                    | **Risk Level** | {risk_level} |
                    | **Model Used** | {selected_model_name} |
                    
                    ---
                    
                    {risk_message}
                """)
            
            # ========================================
            # RECOMMENDATIONS
            # ========================================
            
            st.divider()
            st.subheader("Medical Recommendations")
            
            if prediction == 1:
                st.error("""
                    ### High Risk Patient - Recommended Actions:
                    
                    1. **Immediate Consultation:** Schedule an appointment with a cardiologist
                    2. **Lifestyle Changes:**
                       - Adopt a heart-healthy diet (low sodium, low fat)
                       - Increase physical activity (with doctor approval)
                       - Quit smoking and limit alcohol
                    3. **Monitoring:** Regular blood pressure and cholesterol checks
                    4. **Medication:** Discuss with doctor about preventive medications
                    5. **Stress Management:** Consider stress-reduction techniques
                    
                    **Disclaimer:** This is a predictive model and should not replace professional medical advice.
                """)
            else:
                st.success("""
                    ### Low Risk Patient - Preventive Measures:
                    
                    1. **Maintain Healthy Habits:**
                       - Continue balanced diet
                       - Regular exercise (150 min/week)
                       - Maintain healthy weight
                    2. **Regular Check-ups:** Annual physical examinations
                    3. **Monitor:** Keep track of blood pressure and cholesterol
                    4. **Prevention:** Stay informed about heart health
                    5. **Family History:** Be aware of family heart disease history
                    
                     **Keep it up!** Continue maintaining a healthy lifestyle.
                """)
            
            # ========================================
            # FEATURE IMPORTANCE
            # ========================================
            
            st.divider()
            st.subheader("Feature Analysis")
            
            # Show which features contributed most
            feature_names = [
                'Age', 'Sex', 'Chest Pain', 'BP', 'Cholesterol',
                'Blood Sugar', 'ECG', 'Max Heart Rate', 'Angina',
                'ST Depression', 'Slope', 'Vessels', 'Thalassemia'
            ]
            
            feature_values = features[0]
            
            # Create bar chart of input features
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=feature_values,
                    marker_color='indianred',
                    text=feature_values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Patient Feature Values',
                xaxis_title='Features',
                yaxis_title='Values',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ========================================
            # COMPARISON WITH DATASET
            # ========================================
            
            st.divider()
            st.subheader("Comparison with Dataset")
            
            df = load_data()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_age = df['age'].mean()
                st.metric(
                    "Age vs Average",
                    f"{age} years",
                    f"{age - avg_age:.1f} years",
                    delta_color="off"
                )
            
            with col2:
                avg_chol = df['chol'].mean()
                st.metric(
                    "Cholesterol vs Average",
                    f"{chol} mg/dl",
                    f"{chol - avg_chol:.0f} mg/dl",
                    delta_color="inverse"
                )
            
            with col3:
                avg_thalach = df['thalach'].mean()
                st.metric(
                    "Max Heart Rate vs Average",
                    f"{thalach} bpm",
                    f"{thalach - avg_thalach:.0f} bpm",
                    delta_color="normal"
                )
    
    else:
        st.info("Adjust the patient parameters above and click 'PREDICT RISK' to get results")
    
    # ========================================
    # SAVED PREDICTIONS
    # ========================================
    
    st.divider()
    
    with st.expander("Save & Export Results"):
        st.markdown("""
            ### Export Prediction Results
            
            You can save the prediction results for medical records or further analysis.
        """)
        
        if predict_button:
            # Create results dictionary
            results_data = {
                'Patient Info': {
                    'Age': age,
                    'Sex': 'Male' if sex == 1 else 'Female',
                    'Chest Pain Type': cp,
                },
                'Prediction': {
                    'Result': 'Disease' if prediction == 1 else 'No Disease',
                    'Probability': f"{probability:.2%}",
                    'Risk Level': risk_level,
                    'Model': selected_model_name
                }
            }
            
            st.json(results_data)
            
            # Download button
            results_df = pd.DataFrame([{
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
                'thal': thal, 'prediction': prediction, 'probability': probability,
                'model': selected_model_name
            }])
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.title("Live Prediction")
    
    st.info("""
        Make real-time heart disease risk predictions 
        using trained ML models.
    """)
    
    st.divider()
    
    st.subheader(" How to Use")
    st.markdown("""
        1. Select a ML model
        2. Enter patient parameters
        3. Click 'Predict Risk'
        4. Review results and recommendations
    """)
    
    st.divider()
    
    st.subheader("Parameter Guide")
    st.markdown("""
        **Age:** 30-80 years  
        **BP:** 90-200 mm Hg  
        **Cholesterol:** 120-400 mg/dl  
        **Max HR:** 70-200 bpm
    """)
    
    st.divider()
    
    st.warning("""
        **Medical Disclaimer**
        
        This tool is for educational purposes only. 
        Always consult with qualified healthcare 
        professionals for medical decisions.
    """)

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()