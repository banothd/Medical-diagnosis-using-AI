import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_option_menu import option_menu
import time
import plotly.express as px
import plotly.graph_objects as go

# Change Name & Logo
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

# Hiding Streamlit add-ons
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Adding Background Image
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.7);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px #000000;
    }
    .sub-header {
        font-size: 30px;
        color: #4CAF50;
        margin-bottom: 10px;
    }
    .info-text {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 5px;
        color: white;
    }
    .prediction-container {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Create session state for storing history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}

# Load the saved models
@st.cache_resource
def load_models():
    models = {
        'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
        'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
        'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
        'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
        'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
    }
    return models

models = load_models()

# Navigation sidebar with option_menu
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white;'>Health Guardian</h1>", unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title="Disease Prediction",
        options=[
            "Home", 
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Parkinsons Prediction",
            "Lung Cancer Prediction",
            "Hypo-Thyroid Prediction",
            "Prediction History",
            "About"
        ],
        icons=["house", "activity", "heart", "person", "lungs", "clipboard2-pulse", "clock-history", "info-circle"],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "rgba(0,0,0,0.5)"},
            "icon": {"color": "white", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4CAF50"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='text-align: center; color: white;'>Powered by ML</div>", unsafe_allow_html=True)

def display_input(label, tooltip, key, min_value=None, max_value=None, type="text"):
    with st.container():
        if type == "text":
            return st.text_input(label, key=key, help=tooltip)
        elif type == "number":
            return st.number_input(label, key=key, help=tooltip, step=1, min_value=min_value, max_value=max_value)
        elif type == "slider":
            return st.slider(label, min_value=min_value, max_value=max_value, key=key, help=tooltip)
        elif type == "radio":
            options = [0, 1]
            option_labels = ["No", "Yes"] if "Yes" in tooltip else ["Female", "Male"]
            return st.radio(label, options=range(len(options)), format_func=lambda x: option_labels[x], key=key, help=tooltip, horizontal=True)

def show_disease_info(disease_name, symptoms, risk_factors, prevention_tips):
    with st.expander("Learn More About This Disease", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Common Symptoms")
            for symptom in symptoms:
                st.markdown(f"- {symptom}")
        
        with col2:
            st.markdown("### Risk Factors")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        
        st.markdown("### Prevention Tips")
        for tip in prevention_tips:
            st.markdown(f"- {tip}")

def show_prediction_progress():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        if i < 30:
            status_text.text("Analyzing input data...")
        elif i < 60:
            status_text.text("Processing through model...")
        elif i < 90:
            status_text.text("Finalizing results...")
        else:
            status_text.text("Prediction complete!")
        time.sleep(0.02)
    
    status_text.empty()
    progress_bar.empty()

def save_to_history(disease_type, inputs, result):
    if disease_type not in st.session_state.prediction_history:
        st.session_state.prediction_history[disease_type] = []
    
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_history[disease_type].append({
        "timestamp": timestamp,
        "inputs": inputs,
        "result": result
    })

# Home Page
if selected == "Home":
    st.markdown("<h1 class='main-header'>Welcome to Health Guardian</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='info-text'>", unsafe_allow_html=True)
        st.markdown("""
        ## Your Personal Health Predictor
        
        This application uses machine learning to predict the likelihood of various diseases based on your health parameters. Our system can help with early detection of:
        
        - **Diabetes**
        - **Heart Disease**
        - **Parkinson's Disease**
        - **Lung Cancer**
        - **Hypo-Thyroid Disease**
        
        **How to use:**
        1. Select a disease category from the sidebar
        2. Input your health parameters
        3. Get instant predictions
        4. View your prediction history
        
        **Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Mock stats with animated counters
        st.markdown("<div style='background-color:rgba(255,255,255,0.1); padding:20px; border-radius:10px;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:white;'>Statistics</h3>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label="Predictions Made", value="5,000+", delta="↑ 12%")
            st.metric(label="Disease Types", value="5", delta="↑ 2")
        with col_b:
            st.metric(label="Accuracy", value="95%", delta="↑ 3%")
            st.metric(label="Users Helped", value="2,500+", delta="↑ 15%")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add quick access buttons
    st.markdown("<h3 style='color:white; margin-top:30px;'>Quick Access</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("Diabetes Test"):
            st.session_state.navigate_to = "Diabetes Prediction"
            st.experimental_rerun()
    with col2:
        if st.button("Heart Test"):
            st.session_state.navigate_to = "Heart Disease Prediction"
            st.experimental_rerun()
    with col3:
        if st.button("Parkinson's Test"):
            st.session_state.navigate_to = "Parkinsons Prediction"
            st.experimental_rerun()
    with col4:
        if st.button("Lung Cancer Test"):
            st.session_state.navigate_to = "Lung Cancer Prediction"
            st.experimental_rerun()
    with col5:
        if st.button("Thyroid Test"):
            st.session_state.navigate_to = "Hypo-Thyroid Prediction"
            st.experimental_rerun()

# Diabetes Prediction Page
elif selected == 'Diabetes Prediction':
    st.markdown("<h1 class='main-header'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    
    # Add informational content
    diabetes_symptoms = ["Increased thirst", "Frequent urination", "Extreme hunger", "Unexplained weight loss", "Fatigue", "Blurred vision"]
    diabetes_risk_factors = ["Family history", "Age (45+)", "Obesity", "Physical inactivity", "Gestational diabetes", "Polycystic ovary syndrome"]
    diabetes_prevention = ["Maintain healthy weight", "Regular physical activity", "Eat healthy foods", "Control blood pressure and cholesterol"]
    
    show_disease_info("Diabetes", diabetes_symptoms, diabetes_risk_factors, diabetes_prevention)
    
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    st.write("Enter the following details to predict diabetes:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies', min_value=0, max_value=20, type="number")
        Glucose = display_input('Glucose Level (mg/dL)', 'Enter glucose level', 'Glucose', min_value=0, max_value=500, type="slider")
        BloodPressure = display_input('Blood Pressure (mm Hg)', 'Enter blood pressure value', 'BloodPressure', min_value=0, max_value=200, type="slider")
        SkinThickness = display_input('Skin Thickness (mm)', 'Enter skin thickness value', 'SkinThickness', min_value=0, max_value=100, type="number")
    
    with col2:
        Insulin = display_input('Insulin Level (mu U/ml)', 'Enter insulin level', 'Insulin', min_value=0, max_value=900, type="number")
        BMI = display_input('BMI value', 'Enter Body Mass Index value', 'BMI', min_value=0.0, max_value=70.0, type="number")
        DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter diabetes pedigree function value', 'DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, type="number")
        Age = display_input('Age', 'Enter age of the person', 'Age', min_value=0, max_value=120, type="number")
    
    diab_diagnosis = ''
    diab_probability = 0.0
    
    if st.button('Diabetes Test Result'):
        show_prediction_progress()
        
        inputs = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure, 
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        }
        
        # Get prediction probability
        diab_prediction_prob = models['diabetes'].predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_probability = diab_prediction_prob[0][1] * 100  # Probability of having diabetes
        
        diab_prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
        
        # Save to history
        save_to_history("Diabetes", inputs, diab_diagnosis)
        
        # Display result with visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if diab_prediction[0] == 1:
                st.error(f"🚨 {diab_diagnosis}")
                st.markdown(f"**Confidence: {diab_probability:.1f}%**")
                st.markdown("""
                **Recommendation:**
                - Consult with a healthcare provider as soon as possible
                - Monitor blood sugar levels regularly
                - Consider lifestyle modifications including diet and exercise
                """)
            else:
                st.success(f"✅ {diab_diagnosis}")
                st.markdown(f"**Confidence: {100-diab_probability:.1f}%**")
                st.markdown("""
                **Recommendation:**
                - Continue maintaining a healthy lifestyle
                - Regular check-ups are still important
                - Monitor blood sugar levels periodically
                """)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = diab_probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "limegreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization
        st.markdown("### Key Factors Analysis")
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        feature_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        # Normal ranges
        normal_ranges = {
            'Pregnancies': [0, 10],
            'Glucose': [70, 100],
            'BloodPressure': [90, 120],
            'SkinThickness': [12, 25],
            'Insulin': [16, 166],
            'BMI': [18.5, 24.9],
            'DiabetesPedigreeFunction': [0.0, 0.5],
            'Age': [0, 120]
        }
        
        # Calculate if values are outside normal range
        status = []
        for i, feat in enumerate(features):
            if feature_values[i] < normal_ranges[feat][0]:
                status.append("Low")
            elif feature_values[i] > normal_ranges[feat][1]:
                status.append("High")
            else:
                status.append("Normal")
        
        # Create DataFrame for visualization
        df_features = pd.DataFrame({
            'Feature': features,
            'Value': feature_values,
            'Status': status
        })
        
        # Color mapping
        color_map = {'Low': 'blue', 'Normal': 'green', 'High': 'red'}
        df_features['Color'] = df_features['Status'].map(color_map)
        
        fig = px.bar(df_features, x='Feature', y='Value', color='Status', 
                     color_discrete_map=color_map,
                     title='Your Health Parameters vs. Normal Range')
        
        # Add normal range as rectangles
        for i, feat in enumerate(features):
            fig.add_shape(type="rect",
                x0=i-0.4, y0=normal_ranges[feat][0],
                x1=i+0.4, y1=normal_ranges[feat][1],
                line=dict(color="green", width=2),
                fillcolor="rgba(0,255,0,0.2)")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.markdown("<h1 class='main-header'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
    
    # Add informational content
    heart_symptoms = ["Chest pain or discomfort", "Shortness of breath", "Pain or discomfort in arms or shoulder", "Fatigue", "Lightheadedness, dizziness or fainting", "Swelling in the legs, ankles or feet"]
    heart_risk_factors = ["Age (Men>45, Women>55)", "Family history", "Smoking", "High blood pressure", "High cholesterol", "Diabetes", "Obesity", "Physical inactivity"]
    heart_prevention = ["Don't smoke", "Exercise regularly", "Maintain a healthy diet", "Maintain a healthy weight", "Get regular health screenings", "Manage stress"]
    
    show_disease_info("Heart Disease", heart_symptoms, heart_risk_factors, heart_prevention)
    
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    st.write("Enter the following details to predict heart disease:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = display_input('Age', 'Enter age of the person', 'age', min_value=1, max_value=120, type="number")
        sex = display_input('Sex', 'Male or Female', 'sex', type="radio")
        cp = display_input('Chest Pain Type', '0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic', 'cp', min_value=0, max_value=3, type="slider")
        trestbps = display_input('Resting Blood Pressure (mm Hg)', 'Enter resting blood pressure', 'trestbps', min_value=80, max_value=200, type="slider")
        chol = display_input('Serum Cholesterol (mg/dl)', 'Enter serum cholesterol', 'chol', min_value=100, max_value=600, type="slider")
    
    with col2:
        fbs = display_input('Fasting Blood Sugar > 120 mg/dl', 'Yes or No', 'fbs', type="radio")
        restecg = display_input('Resting ECG Results', '0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy', 'restecg', min_value=0, max_value=2, type="slider")
        thalach = display_input('Maximum Heart Rate', 'Enter maximum heart rate achieved', 'thalach', min_value=50, max_value=220, type="slider")
        exang = display_input('Exercise Induced Angina', 'Yes or No', 'exang', type="radio")
        oldpeak = display_input('ST Depression Induced by Exercise', 'Enter ST depression value', 'oldpeak', min_value=0.0, max_value=10.0, type="number")
    
    with col3:
        slope = display_input('Slope of Peak Exercise ST Segment', '0: Upsloping, 1: Flat, 2: Downsloping', 'slope', min_value=0, max_value=2, type="slider")
        ca = display_input('Number of Major Vessels Colored by Fluoroscopy', 'Enter number (0-3)', 'ca', min_value=0, max_value=3, type="slider")
        thal = display_input('Thalassemia', '0: Normal, 1: Fixed Defect, 2: Reversible Defect', 'thal', min_value=0, max_value=2, type="slider")
    
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        show_prediction_progress()
        
        inputs = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Get prediction probability
        heart_prediction_prob = models['heart_disease'].predict_proba([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_probability = heart_prediction_prob[0][1] * 100  # Probability of having heart disease
        
        heart_prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        
        # Save to history
        save_to_history("Heart Disease", inputs, heart_diagnosis)
        
        # Display result with visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if heart_prediction[0] == 1:
                st.error(f"🚨 {heart_diagnosis}")
                st.markdown(f"**Confidence: {heart_probability:.1f}%**")
                st.markdown("""
                **Recommendation:**
                - Consult with a cardiologist as soon as possible
                - Monitor blood pressure and cholesterol regularly
                - Consider lifestyle modifications including diet, exercise, and stress management
                """)
            else:
                st.success(f"✅ {heart_diagnosis}")
                st.markdown(f"**Confidence: {100-heart_probability:.1f}%**")
                st.markdown("""
                **Recommendation:**
                - Continue maintaining a heart-healthy lifestyle
                - Regular check-ups are still important
                - Monitor blood pressure and cholesterol periodically
                """)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = heart_probability,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Heart Disease Risk"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 20], 'color': "green"},
                        {'range': [20, 40], 'color': "limegreen"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        # Heart health visualization
        st.markdown("### Heart Health Analysis")
        
        # Create radar chart for key heart health indicators
        categories = ['Blood Pressure', 'Cholesterol', 'Heart Rate', 'Exercise Tolerance', 'Vessel Health']
        
        # Normalize values to 0-100 scale for radar chart
        bp_score = 100 - min(100, max(0, (trestbps - 90) * 100 / 110))  # Lower is better (ideal 90-120)
        chol_score = 100 - min(100, max(0, (chol - 150) * 100 / 300))   # Lower is better (ideal 150-200)
        hr_score = min(100, max(0, (220 - age - abs(thalach - (220 - age) * 0.85)) * 100 / 50))  # Closer to 85% of max is better
        ex_score = 100 - min(100, max(0, exang * 50 + oldpeak * 10))    # Lower is better
        vessel_score = 100 - min(100, max(0, ca * 33.3))                # Lower is better
        
        values = [bp_score, chol_score, hr_score, ex_score, vessel_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Health',
            line_color='rgb(111, 231, 219)',
            fillcolor='rgba(111, 231, 219, 0.5)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[80, 80, 80, 80, 80],
            theta=categories,
            fill='toself',
            name='Good Health',
            line_color='rgba(0, 200, 0, 0.8)',
            fillcolor='rgba(0, 200, 0, 0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Parkinson's Prediction Page
elif selected == "Parkinsons Prediction":
    st.markdown("<h1 class='main-header'>Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
    
    # Add informational content
    parkinsons_symptoms = ["Tremor", "Slowed movement (bradykinesia)", "Rigid muscles", "Impaired posture and balance", "Loss of automatic movements", "Speech changes", "Writing changes"]
    parkinsons_risk_factors = ["Age (60+)", "Heredity", "Sex (men are more likely)", "Exposure to toxins", "Serious head injury"]
    parkinsons_prevention = ["Regular exercise", "Healthy diet", "Avoiding exposure to pesticides and toxins", "Regular check-ups"]
    
    show_disease_info("Parkinson's Disease", parkinsons_symptoms, parkinsons_risk_factors, parkinsons_prevention)
    
    st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
    
    # Simplified interface for parkinsons with three tabs
    tab1, tab2, tab3 = st.tabs(["Voice Parameters", "Frequency Variations", "Additional Measurements"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fo = display_input('MDVP:Fo(Hz) - Average vocal fundamental frequency', 'Enter MDVP:Fo(Hz) value', 'fo', min_value=80.0, max_value=260.0, type="number")
            fhi = display_input('MDVP:Fhi(Hz) - Maximum vocal fundamental frequency', 'Enter MDVP:Fhi(Hz) value', 'fhi', min_value=100.0, max_value=600.0, type="number")
            flo = display_input('MDVP:Flo(Hz) - Minimum vocal fundamental frequency', 'Enter MDVP:Flo(Hz) value', 'flo', min_value=60.0, max_value=240.0, type="number")
        with col2:
            Jitter_percent = display_input('MDVP:Jitter(%) - Variation in fundamental frequency', 'Enter MDVP:Jitter(%) value', 'Jitter_percent', min_value=0.0, max_value=1.0, type="number")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            Jitter_percent = display_input('MDVP:Jitter(%)', 'Enter MDVP:Jitter(%) value', 'jitter_percent', min_value=0.0, max_value=1.0, type="number")
            Jitter_Abs = display_input('MDVP:Jitter(Abs)', 'Enter MDVP:Jitter(Abs) value', 'jitter_abs', min_value=0.0, max_value=0.1, type="number")
            RAP = display_input('MDVP:RAP', 'Enter MDVP:RAP value', 'rap', min_value=0.0, max_value=1.0, type="number")
        with col2:
            PPQ = display_input('MDVP:PPQ', 'Enter MDVP:PPQ value', 'ppq', min_value=0.0, max_value=1.0, type="number")
            DDP = display_input('Jitter:DDP', 'Enter Jitter:DDP value', 'ddp', min_value=0.0, max_value=1.0, type="number")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            Shimmer = display_input('MDVP:Shimmer', 'Enter MDVP:Shimmer value', 'shimmer', min_value=0.0, max_value=1.0, type="number")
            Shimmer_dB = display_input('MDVP:Shimmer(dB)', 'Enter MDVP:Shimmer(dB) value', 'shimmer_db', min_value=0.0, max_value=10.0, type="number")
            APQ3 = display_input('Shimmer:APQ3', 'Enter Shimmer:APQ3 value', 'apq3', min_value=0.0, max_value=1.0, type="number")
        with col2:
            APQ5 = display_input('Shimmer:APQ5', 'Enter Shimmer:APQ5 value', 'apq5', min_value=0.0, max_value=1.0, type="number")
            APQ = display_input('MDVP:APQ', 'Enter MDVP:APQ value', 'apq', min_value=0.0, max_value=1.0, type="number")
            DDA = display_input('Shimmer:DDA', 'Enter Shimmer:DDA value', 'dda', min_value=0.0, max_value=1.0, type="number")
    
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        show_prediction_progress()
        
        inputs = {
            'fo': fo,
            'fhi': fhi,
            'flo': flo,
            'jitter_percent': Jitter_percent,
            'jitter_abs': Jitter_Abs,
            'rap': RAP,
            'ppq': PPQ,
            'ddp': DDP,
            'shimmer': Shimmer,
            'shimmer_db': Shimmer_dB,
            'apq3': APQ3,
            'apq5': APQ5,
            'apq': APQ,
            'dda': DDA
        }
        
        parkinsons_prediction = models['parkinsons'].predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA]])
        
        parkinsons_diagnosis = "The person has Parkinson's disease' if parkinsons_prediction[0] == 1 else 'The person does not have Parkinson's disease"
        
        # Save to history
        save_to_history("Parkinson's Disease", inputs, parkinsons_diagnosis)
        
        if parkinsons_prediction[0] == 1:
            st.error(f"🚨 {parkinsons_diagnosis}")
            st.markdown("""
            **Recommendation:**
            - Consult a neurologist
            - Follow prescribed medications and therapies
            - Engage in regular exercise and speech therapy
            - Monitor symptoms closely
            """)
        else:
            st.success(f"✅ {parkinsons_diagnosis}")
            st.markdown("""
            **Recommendation:**
            - Maintain a healthy lifestyle
            - Regular health check-ups
            - Stay physically and mentally active
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)
