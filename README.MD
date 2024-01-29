# Stroke Prediction Web App

## Overview
This is a web application that predicts the risk of stroke based on user input.


## Install Requirements
Before running the app, make sure you have the following Python packages installed:
- `streamlit==1.30.0`
- `joblib==1.3.2`
- `pandas==1.5.3`
- `scikit-learn==1.2.2`
- `numpy==1.23.5`

You can install these packages using pip with the following command:
```bash
pip install -r requirements.txt
```

## Usage
- **Running the App**:
  - To run the app, use the following command:
    ```bash
    streamlit run app.py
    ```
  -  - Open your web browser and access the app at [this link](https://stroke-prediction-web-app-bwappucx4qt8k5nqeg2x9yh.streamlit.app/).


- **Age**: Enter the age of the patient (between 0 and 80 years).
- **Gender**: Select the gender of the patient (Female or Male).
- **BMI (Body Mass Index)**: Enter the BMI of the patient (between 10 and 100).
- **Average Glucose Level**: Enter the average glucose level of the patient (between 50 and 300).
- **Hypertension**: Select whether the patient has hypertension (Yes or No).
- **Heart Disease**: Select whether the patient has heart disease (Yes or No).
- **Smoking Status**: Select the smoking status of the patient (Never Smoked, Smoker, Formerly Smoked, Prefer Not To Say).
- **Work Type**: Select the type of work (Private, Self Employed, Government, Child, Never Worked).
- **Ever Married**: Select whether the patient has ever been married (Yes or No).
