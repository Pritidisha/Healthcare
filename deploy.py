import pandas as pd
import streamlit as st
import joblib
import sklearn

model = joblib.load('health_gbr.joblib')

def predict_charges(model, age, sex, bmi, children, smoker, region):
    user_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
 
    preprocessor = model.named_steps['preprocessor']
    transformed_data = preprocessor.transform(user_data)

    prediction = model.named_steps['regressor'].predict(transformed_data)[0]
    return prediction

st.set_page_config(page_title="Health Insurance Charges Prediction")
st.title('Health Insurance Charges Prediction')

age = st.number_input('Age:', min_value=0)
sex = st.selectbox('Sex:', ['male', 'female'])
bmi = st.number_input('BMI:', min_value=0.0)
children = st.number_input('Number of Children:', min_value=0)
smoker = st.selectbox('Smoker:', ['no', 'yes'])
region = st.selectbox('Region:', ['northeast', 'northwest', 'southeast', 'southwest'])

if st.button('Predict Charges'): 
    predicted_charges = predict_charges(model, age, sex, bmi, children, smoker, region)
    st.success(f"Predicted Insurance Charges: {predicted_charges:.2f}")
