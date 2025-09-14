# app.py

# --- IMPORTANT ---
# To run this app, make sure 'prediction.py' and your image/model files
# are in the SAME FOLDER as this script. This will solve the import error.
# -----------------

import streamlit as st
from predictions import make_prediction # This line imports the function from prediction.py
import base64
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to set background image ---
def set_bg_from_local(file_path):
    """
    Sets a background image from a local file.
    Args:
        file_path (str): The path to the image file.
    """
    try:
        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        # Custom CSS to inject the background image and style elements
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Sidebar style adjustments */
            [data-testid="stSidebar"] > div:first-child {{
                background-color: rgba(0, 0, 0, 0.8); /* Semi-transparent black background */
                color: white; /* Sets the default text color to white */
            }}
            [data-testid="stSidebar"] h1 {{
                color: white; /* Ensures the sidebar title is white */
            }}
            .main .block-container {{
                 background-color: rgba(0, 0, 0, 0.6); /* dark transparent background */
                 color: white; /* ensures all text is readable */
                 padding: 2rem;
                 border-radius: 10px;
                 box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.5);
            }}
            h1, h2, h3 {{
                color: #003366; /* A professional dark blue color */
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error("Background image not found. Make sure 'background.jpg' is in the same folder.")

# Set the background image
set_bg_from_local('background.jpg')


# --- Page Navigation in Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Prediction Dashboard", "Project Analysis & Insights"])


# --- PREDICTION DASHBOARD PAGE ---
if page == "Prediction Dashboard":
    st.title("🔬 Thyroid Cancer Recurrence Predictor")
    st.markdown("""
    This app uses a fine-tuned Random Forest model to predict the likelihood of thyroid cancer recurrence.
    Please input the patient's information using the sidebar to receive a prediction.
    """)
    st.markdown("---")

    # --- Input Fields for User in the Sidebar ---
    st.sidebar.header("Patient Information Input")

    age = st.sidebar.slider("Age at Diagnosis", 15, 82, 40)
    gender = st.sidebar.selectbox("Gender", ["F", "M"])
    smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
    hx_smoking = st.sidebar.selectbox("History of Smoking", ["No", "Yes"])
    hx_radiotherapy = st.sidebar.selectbox("History of Radiotherapy", ["No", "Yes"])
    thyroid_function = st.sidebar.selectbox("Thyroid Function", ['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'])
    physical_exam = st.sidebar.selectbox("Physical Examination", ['Single nodular goiter-left', 'Multinodular goiter', 'Single nodular goiter-right', 'Normal', 'Diffuse goiter'])
    adenopathy = st.sidebar.selectbox("Adenopathy", ['No', 'Right', 'Extensive', 'Bilateral', 'Left', 'Central'])
    pathology = st.sidebar.selectbox("Pathology", ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
    focality = st.sidebar.selectbox("Focality", ['Uni-Focal', 'Multi-Focal'])
    risk = st.sidebar.selectbox("Risk Category", ['Low', 'Intermediate', 'High'])
    t_stage = st.sidebar.selectbox("T Stage", ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
    n_stage = st.sidebar.selectbox("N Stage", ['N0', 'N1b', 'N1a'])
    m_stage = st.sidebar.selectbox("M Stage", ['M0', 'M1'])
    stage = st.sidebar.selectbox("Overall Stage", ['I', 'II', 'IVB', 'III', 'IVA'])
    response = st.sidebar.selectbox("Response to Treatment", ['Indeterminate', 'Excellent', 'Structural Incomplete', 'Biochemical Incomplete'])

    # --- Prediction Button and Display Logic ---
    if st.sidebar.button("Predict Recurrence", key="predict_button"):
        input_features = {
            'Age': age, 'Gender': gender, 'Smoking': smoking, 'Hx Smoking': hx_smoking,
            'Hx Radiothreapy': hx_radiotherapy, # This key must match the column name your model was trained on
            'Thyroid Function': thyroid_function,
            'Physical Examination': physical_exam, 'Adenopathy': adenopathy,
            'Pathology': pathology, 'Focality': focality, 'Risk': risk,
            'T': t_stage, 'N': n_stage, 'M': m_stage, 'Stage': stage,
            'Response': response
        }
        
        prediction, probability = make_prediction(input_features)
        
        st.header("Prediction Result")
        if prediction == 1:
            st.error(f"**High Risk of Recurrence**")
            st.progress(float(probability))
            st.metric(label="Probability of Recurrence", value=f"{probability*100:.2f}%")
        else:
            st.success(f"**Low Risk of Recurrence**")
            st.progress(float(probability))
            st.metric(label="Probability of Recurrence", value=f"{probability*100:.2f}%")
        
        st.info("Disclaimer: This prediction is based on a machine learning model and is for informational purposes only. It is not a substitute for professional medical advice.")

# --- PROJECT ANALYSIS PAGE ---
elif page == "Project Analysis & Insights":
    st.title("📊 Project Analysis & Model Insights")
    st.markdown("This page showcases the key visualizations from the project, demonstrating the model's high performance and interpretability.")
    
    st.markdown("---")
    
    st.header("Model Performance Evaluation")
    st.markdown("The model was evaluated on unseen test data, achieving 97% accuracy. The plots below show its effectiveness in distinguishing between patients with and without recurrence.")
    
    try:
        # Use the new, descriptive filename for the performance plots
        st.image('performance_plots.png', caption='Evaluation Metrics: Confusion Matrix, ROC Curve, and Precision-Recall Curve')
    except FileNotFoundError:
        st.warning("Performance plot image ('performance_plots.png') not found. Please add it to the folder.")

    st.markdown("---")
    
    st.header("Model Interpretability (SHAP Analysis)")
    st.markdown("""
    To understand *why* the model makes its decisions, I used SHAP (SHapley Additive exPlanations). The plot below reveals the impact of each clinical feature on the model's output, providing crucial transparency.
    """)
    
    try:
        # Use the new, descriptive filename for the SHAP plot
        st.image('shap_plot.png', caption='SHAP Summary Plot: Explaining Feature Impact')
    except FileNotFoundError:
        st.warning("SHAP plot image ('shap_plot.png') not found. Please add it to the folder.")


