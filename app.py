import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Breast Cancer Survival Prediction")

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "svm_pipeline.pkl"), "rb"))

st.title("Breast Cancer Survival Prediction (SVM)")

# Load dataset structure (for inputs)
data = pd.read_csv("Breast_Cancer.csv")
X = data.drop("Status", axis=1)

input_data = {}

st.subheader("Enter Patient Details")

for col in X.columns:
    if X[col].dtype == "object":
        input_data[col] = st.selectbox(col, X[col].unique())
    else:
        input_data[col] = st.number_input(col, float(X[col].min()), float(X[col].max()))

if st.button("Predict Survival Status"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    if prediction == "Alive":
        st.success("🟢 Patient is likely to SURVIVE")
    else:
        st.error("🔴 Patient is NOT likely to survive")
