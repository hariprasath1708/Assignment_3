import streamlit as st
import pickle
import numpy as np
import xgboost

# Load models and scalers
with open('D:/streamlit4/env/Scripts/parkinsons_model1.pkl', 'rb') as f:
    parkinsons_data = pickle.load(f)
parkinsons_scaler = parkinsons_data['scaler']
parkinsons_model = parkinsons_data['model1']

with open('D:/streamlit4/env/Scripts/india_liver_model1.pkl', 'rb') as f:
    liver_data = pickle.load(f)
liver_encoder = liver_data['scaler']
liver_model = liver_data['model1']

with open('D:/streamlit4/env/Scripts/kidney_model6.pkl', 'rb') as f:
    kidney_data = pickle.load(f)
kidney_encoder = kidney_data['scaler']
kidney_model = kidney_data['model6']

# Add custom CSS to style the page
st.markdown("""
    <style>
        /* Global styles */
        body {
            background-color: #f4f8ff; /* Light background color */
            font-family: 'Arial', sans-serif;
            color: #333;
            line-height: 1.6;
        }

        /* Sidebar styles */
        .stSidebar {
            background-color: #007acc; /* Sidebar blue color */
            color: white;
        }

        .stHeader {
            background-color: #4682b4; /* Steel Blue header */
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 90px;
        }

        .stSubheader {
            color: #007acc;
            font-size: 20px;
            font-weight: 600;
        }

        .stButton {
            background-color: #4682b4; 
            color: white;
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 18px;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
        }

        .stButton:hover {
            background-color: #5a9bd5;
            transform: translateY(-2px);
        }

        /* Custom form input styles */
        .stNumberInput {
            margin-top: 10px;
            font-size: 16px;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }

        .stSelectbox {
            margin-bottom: 15px;
            font-size: 16px;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }

        /* Custom result card styles */
        .prediction-result {
            background-color: #e0f7fa; /* Light cyan background */
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        .prediction-result h3 {
            font-size: 24px;
            color: #333;
            font-weight: bold;
        }

        .prediction-result p {
            font-size: 18px;
            color: #4682b4;
        }

        .stMarkdown {
            font-size: 18px;
            line-height: 1.8;
            margin-top: 10px;
        }

        /* Styling the SelectBox and NumberInput inside the Sidebar */
        .sidebar-box {
            background-color: #007acc;
            color: white;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit header
st.header("Multiple Disease Prediction")
st.subheader("Disease Prediction Dashboard")

# Sidebar with disease selection
st.sidebar.markdown("""
    <div class="sidebar-box">
        <h2>Select a Disease to Predict</h2>
    </div>
    """, unsafe_allow_html=True)

selected_dataset = st.sidebar.selectbox(
    "Select a Disease to Predict",
    ("Parkinson's Disease", "Liver Disease", "Kidney Disease")
)

# Disease prediction logic
if selected_dataset == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    # Add all inputs for Parkinson's Disease prediction
    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, step=0.001)
    HNR = st.number_input("HNR", min_value=0.0, step=0.1)
    RPDE = st.number_input("RPDE", min_value=0.0, step=0.001)
    DFA = st.number_input("DFA", min_value=0.0, step=0.001)
    spread1 = st.number_input("Spread1", step=0.001)
    spread2 = st.number_input("Spread2", step=0.001)
    D2 = st.number_input("D2", min_value=0.0, step=0.001)
    PPE = st.number_input("PPE", min_value=0.0, step=0.001)

    input_features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,
                                 MDVP_Shimmer, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

    scaled_features = parkinsons_scaler.transform(input_features)
    if st.button("Predict", key="parkinsons_predict"):
        prediction = parkinsons_model.predict(scaled_features)
        result = "Positive for Parkinson's Disease" if prediction[0] == 1 else "Negative for Parkinson's Disease"
        st.markdown(f"""
            <div class="prediction-result">
                <h3>Prediction Result:</h3>
                <p>{result}</p>
            </div>
        """, unsafe_allow_html=True)

elif selected_dataset == "Liver Disease":
    st.header("Liver Disease Prediction")
    # Add all inputs for Liver Disease prediction
    Age = st.number_input("Age", min_value=1, step=1)
    Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0)
    Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0)
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0)
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0.0)
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0)
    Total_Proteins = st.number_input("Total Proteins", min_value=0.0)
    Albumin = st.number_input("Albumin", min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0)

    input_features = np.array([[Age, Total_Bilirubin, Direct_Bilirubin,
        Alkaline_Phosphotase, Alamine_Aminotransferase,
        Aspartate_Aminotransferase, Total_Proteins, Albumin,
        Albumin_and_Globulin_Ratio]])

    try:
        scaled_features = liver_encoder.transform(input_features)

        if st.button("Predict", key="liver_predict"):
            prediction = liver_model.predict(scaled_features)
            result = "Positive for Liver Disease" if prediction[0] == 1 else "Negative for Liver Disease"
            st.markdown(f"""
                <div class="prediction-result">
                    <h3>Prediction Result:</h3>
                    <p>{result}</p>
                </div>
            """, unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error during prediction: {str(e)}")

elif selected_dataset == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    # Add all inputs for Kidney Disease prediction
    age = st.number_input("Age", min_value=1, step=1)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.030, step=0.001)
    al = st.number_input("Albumin", min_value=0.0)
    su = st.number_input("Sugar", min_value=0.0)
    rbc = st.number_input("Red Blood Cells", min_value=0.0)
    pc = st.number_input("Pus Cells", min_value=0.0)
    sc = st.number_input("Serum Creatinine Level", min_value=0.0)
    sod = st.number_input("Sodium Level in Blood", min_value=0.0)
    wc = st.number_input("White Blood Cells", min_value=0.0)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0)
    cad = st.number_input("Coronary artery Disease", min_value=0.0)
    appet = st.number_input("Appetite Status", min_value=0.0)

    input_features = np.array([[age, bp, sg, al, su, rbc, pc, sc, sod, wc, rc, cad, appet]])
    scaled_features = kidney_encoder.transform(input_features)
    if st.button("Predict", key="kidney_predict"):
        prediction = kidney_model.predict(scaled_features)
        result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
        st.markdown(f"""
            <div class="prediction-result">
                <h3>Prediction Result:</h3>
                <p>{result}</p>
            </div>
        """, unsafe_allow_html=True)
