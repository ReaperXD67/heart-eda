import streamlit as st
import pandas as pd
import joblib

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Heart Disease Prediction By Aman ❤️🙌")
st.markdown("This app predicts whether a person has heart disease or not based on various health parameters. So provide the following details to know the result.")
age = st.slider('Age', 18, 100, 25)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'TA', 'ASY'])
resting_bp = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=80, max_value=200, value=120)
cholestrol = st.number_input('Cholesterol (in mg/dl)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0,1])
resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_heart_rate = st.slider('Max Heart Rate Achieved', 60, 220, 150)
oldpeak = st.slider('Oldpeak', 0.0, 6.0, 1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    raw_data = {
        'Age': age,
        'Sex': sex,
        'Chest Pain Type': chest_pain_type,
        'Resting Blood Pressure': resting_bp,
        'Cholesterol': cholestrol,
        'Fasting Blood Sugar': fasting_bs,
        'Resting ECG': resting_ecg,
        'Max Heart Rate': max_heart_rate,
        'Oldpeak': oldpeak,
        'ST Slope': st_slope
    }

    input_data = pd.DataFrame([raw_data])
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_columns]
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    if prediction == 1:
        st.error('The person is likely to have heart disease. Please consult a doctor for further evaluation.')
    else:
        st.success('The person is unlikely to have heart disease. Maintain a healthy lifestyle!')
