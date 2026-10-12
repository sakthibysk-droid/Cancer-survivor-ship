import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Cancer Classification AI",
    page_icon="🩺",
    layout="wide"
)

st.title("Cancer Classification AI")
st.write(
    "Research demonstration using the Wisconsin Diagnostic "
    "Breast Cancer dataset."
)

st.warning(
    "Research use only. This tool is not a clinical diagnosis "
    "or a validated cancer-risk assessment."
)

# Load the trained model
try:
    model = joblib.load("cancer_model.pkl")
except Exception:
    st.error(
        "Model file not found. Place cancer_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

feature_names = [f"Feature_{i}" for i in range(1, 31)]

st.subheader("Enter the 30 model features")
st.caption(
    "Enter actual feature values from a compatible dataset. "
    "These are computed cell-nucleus measurements, not ordinary "
    "patient symptoms."
)

with st.form("prediction_form"):
    values = {}

    columns = st.columns(3)

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f",
                key=feature
            )

    submitted = st.form_submit_button("Predict classification")

if submitted:
    input_df = pd.DataFrame([values])

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)
    malignant_index = classes.index(1)
    malignant_probability = probabilities[malignant_index]

    if prediction == 1:
        st.error("Model prediction: Malignant")
    else:
        st.success("Model prediction: Benign")

    st.metric(
        "Model-estimated malignant probability",
        f"{malignant_probability:.1%}"
    )

    st.caption(
        "This probability reflects the trained model, not a "
        "confirmed clinical probability for an individual patient."
    )
