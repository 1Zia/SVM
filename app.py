import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "svm_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))

st.set_page_config(page_title="Breast Cancer SVM", layout="centered")
st.title("Breast Cancer Survival Prediction (SVM)")

# ===============================
# User Inputs
# ===============================
input_data = {
    "Age": st.number_input("Age", 1, 120, 50),
    "Race": st.selectbox("Race", ["White", "Black", "Asian", "Other"]),
    "Marital Status": st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"]),
    "T Stage ": st.selectbox("T Stage", ["T1", "T2", "T3", "T4"]),
    "N Stage": st.selectbox("N Stage", ["N0", "N1", "N2", "N3"]),
    "6th Stage": st.selectbox("6th Stage", ["I", "IIA", "IIB", "IIIA", "IIIB", "IIIC"]),
    "differentiate": st.selectbox("Differentiate", ["Well", "Moderate", "Poor"]),
    "Grade": st.selectbox("Grade", ["1", "2", "3"]),
    "A Stage": st.selectbox("A Stage", ["Localized", "Regional", "Distant"]),
    "Tumor Size": st.number_input("Tumor Size", 0, 200, 20),
    "Estrogen Status": st.selectbox("Estrogen Status", ["Positive", "Negative"]),
    "Progesterone Status": st.selectbox("Progesterone Status", ["Positive", "Negative"]),
    "Regional Node Examined": st.number_input("Regional Node Examined", 0, 100, 5),
    "Reginol Node Positive": st.number_input("Regional Node Positive", 0, 50, 1),
    "Survival Months": st.number_input("Survival Months", 0, 200, 60),
}

# ===============================
# Prediction
# ===============================
if st.button("Predict"):
    df = pd.DataFrame([input_data])

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # ✅ ALIGN FEATURES EXACTLY LIKE TRAINING
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)

    if pred[0] == 1:
        st.success("🟢 Patient Survived")
    else:
        st.error("🔴 Patient Did Not Survive")
