import pandas as pd
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Function to load models
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load models
Kidney_model = load_model(r"D:\streamlit4\env\Scripts\kidney.pkl")
liver_model = load_model(r"D:\streamlit4\env\Scripts\liver.pkl")
parkinson_model = load_model(r"D:\streamlit4\env\Scripts\parkinsons_model.pkl")


# Sidebar Navigation
with st.sidebar:
    selected_model = option_menu(
        "Multiple Disease Prediction System 🏥",
        ["Home", "Kidney Prediction", "Liver Prediction", "Parkinsons Prediction"],
        icons=['house', 'activity', 'heart', 'person'],
        menu_icon="hospital-fill",
        default_index=0
    )

# Home Page
if selected_model == "Home":
    st.title("Welcome to the Multiple Disease Prediction System 🏥")
    st.write("""
        This tool helps you predict the risk of various diseases with AI models. 🚀  
        
        - **KIDNEY DISEASE** 💧  
        - **LIVER DISEASE** 🍂  
        - **PARKINSON'S DISEASE** 🧠  
        
        🛠️ **HOW TO USE**:  
        Select a model from the **SIDEBAR MENU** 📂 and input your data.  
        
        💡 Stay proactive about your health! 🌿  
    """)

# Function to display results
def display_result(condition_name, prediction, advice):
    if prediction == 1:
        st.error(f"⚠️ High risk of {condition_name}. 😔")
        st.write(f"💡 **Advice**: {advice}")
    else:
        st.success(f"✅ No significant risk of {condition_name}! 🎉")
        st.write(f"💡 **Advice**: {advice}")

# 🚑 **LIVER DISEASE PREDICTION**
if selected_model == "Liver Prediction":
    st.title("Liver Disease Prediction 🍂")

    # Input fields
    Age = st.number_input('Age', min_value=1)
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Gender = 1 if Gender == 'Male' else 0  # Encode gender
    Total_Bilirubin = st.number_input('Total Bilirubin', min_value=0.0)
    Direct_Bilirubin = st.number_input('Direct Bilirubin', min_value=0.0)
    Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase', min_value=0)
    Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0)
    Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0)
    Total_Proteins = st.number_input('Total Proteins', min_value=0.0)
    Albumin = st.number_input('Albumin', min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0)

    if st.button('Predict'):
        input_data = pd.DataFrame([{
            "Age": Age, "Gender": Gender, "Total_Bilirubin": Total_Bilirubin,
            "Direct_Bilirubin": Direct_Bilirubin, "Alkaline_Phosphotase": Alkaline_Phosphotase,
            "Alamine_Aminotransferase": Alamine_Aminotransferase,
            "Aspartate_Aminotransferase": Aspartate_Aminotransferase, "Total_Proteins": Total_Proteins,
            "Albumin": Albumin, "Albumin_and_Globulin_Ratio": Albumin_and_Globulin_Ratio
        }])

        # Ensure feature names match the trained model
        expected_features = liver_model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        prediction = liver_model.predict(input_data)[0]
        advice = "Maintain a healthy diet with minimal alcohol and fatty foods."
        display_result("Liver Disease", prediction, advice)

# 🚰 **KIDNEY DISEASE PREDICTION**
elif selected_model == "Kidney Prediction":
    st.title("Kidney Disease Prediction 💧")

    # Input fields
    features = {
        "age": st.number_input('Age', min_value=1, max_value=120),
        "bp": st.number_input('Blood Pressure', min_value=0.0),
        "sg": st.number_input('Specific Gravity', min_value=1.000, max_value=1.050, step=0.001, format="%.3f"),
        "al": st.number_input('Albumin', min_value=0),
        "su": st.number_input('Sugar', min_value=0),
        "bgr": st.number_input('Blood Glucose Random', min_value=0.0),
        "bu": st.number_input('Blood Urea', min_value=0.0),
        "sc": st.number_input('Serum Creatinine', min_value=0.0),
    }

    if st.button("Predict"):
        input_data = pd.DataFrame([features])

        # Ensure feature names match the trained model
        expected_features = Kidney_model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        prediction = Kidney_model.predict(input_data)[0]
        advice = "Drink plenty of water and reduce sodium intake."
        display_result("Kidney Disease", prediction, advice)

# 🧠 **PARKINSON'S DISEASE PREDICTION**
elif selected_model == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction 🧠")

    # Input fields
    features = {
        "MDVP:Fo(Hz)": st.number_input('MDVP:Fo(Hz)', min_value=0.0),
        "MDVP:Fhi(Hz)": st.number_input('MDVP:Fhi(Hz)', min_value=0.0),
        "MDVP:Flo(Hz)": st.number_input('MDVP:Flo(Hz)', min_value=0.0),
        "MDVP:Jitter(%)": st.number_input('MDVP:Jitter(%)', min_value=0.0),
        "MDVP:Jitter(Abs)": st.number_input('MDVP:Jitter(Abs)', min_value=0.0),
        "MDVP:RAP": st.number_input('MDVP:RAP', min_value=0.0),
        "MDVP:PPQ": st.number_input('MDVP:PPQ', min_value=0.0),
        "Jitter:DDP": st.number_input('Jitter:DDP', min_value=0.0),
        "MDVP:Shimmer": st.number_input('MDVP:Shimmer', min_value=0.0),
        "HNR": st.number_input('HNR', min_value=0.0),
        "RPDE": st.number_input('RPDE', min_value=0.0),
        "DFA": st.number_input('DFA', min_value=0.0),
        "spread1": st.number_input('spread1', min_value=-100.0, max_value=100.0),
        "spread2": st.number_input('spread2', min_value=0.0),
        "D2": st.number_input('D2', min_value=0.0),
        "PPE": st.number_input('PPE', min_value=0.0),
    }

    if st.button('Predict'):
        input_data = pd.DataFrame([features])

        # Ensure columns match trained model
        expected_features = parkinson_model.feature_names_in_
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

        prediction = parkinson_model.predict(input_data)[0]
        advice = "Exercise regularly and eat a balanced diet."
        display_result("Parkinson’s Disease", prediction, advice)