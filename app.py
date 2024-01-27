import streamlit as st
from joblib import load
import pandas as pd


def load_model(path):
    return load(path)


def user_input():
    st.header("Patient Medical Record")
    st.write("Please enter your details below")

    age = st.number_input("Age", min_value=0.0, max_value=90.0, step=1.0)
    gender = st.selectbox("Gender", options=("Female", "Male"))
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, step=1.0)
    avg_glucose_level = st.number_input(
        "Average Glucose Level", min_value=50.0, max_value=300.0, step=1.0
    )
    hypertension = (
        st.selectbox("Do you have hypertension?", options=("No", "Yes")) == "Yes"
    )
    heart_disease = (
        st.selectbox("Do you have heart disease?", options=("No", "Yes")) == "Yes"
    )
    smoking_status = st.selectbox(
        "What is your smoking status?",
        options=("Never Smoked", "Smoker", "Formerly Smoked", "Prefer Not To Say"),
    )
    work_type = st.selectbox(
        "What is your type of work?",
        options=("Private", "Self Employed", "Government", "Child", "Never Worked"),
    )
    ever_married = st.selectbox("Have you ever been married?", options=("No", "Yes"))

    return pd.DataFrame(
        [
            {
                "age": age,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "ever_married": ever_married,
                "work_type": work_type,
                "smoking_status": smoking_status,
                "hypertension": 1 if hypertension else 0,
                "heart_disease": 1 if heart_disease else 0,
            }
        ]
    )


def predict_stroke(model, input_data):
    prediction_proba = model.predict_proba(input_data)[:, -1]
    return prediction_proba >= 0.56


def main():
    st.title("Stroke Prediciton")

    rf_model = load_model("model/rf_final_pipeline.joblib")

    user_data = user_input()
    if st.button("Predict Stroke Risk"):
        high_risk = predict_stroke(rf_model, user_data)
        if high_risk:
            st.error("The model predicts the patient is at high risk of stroke.")
        else:
            st.success("The model predicts the patient is at low risk of stroke.")


if __name__ == "__main__":
    main()
