import pandas as pd
import streamlit as st
import pickle
import sklearn

# Load the saved model
with open('health_gbr.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_charges(model, age, sex, bmi, children, smoker, region):
    # Preprocess user input data (one-hot encoding for categorical features)
    user_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })
    
    # Transform data using the defined preprocessor from your original script
    preprocessor = model.named_steps['preprocessor']
    transformed_data = preprocessor.transform(user_data)
    
    # Make prediction using the model
    prediction = model.named_steps['regressor'].predict(transformed_data)[0]
    return prediction

st.set_page_config(page_title="Insurance Charges Prediction")
st.title('Insurance Charges Prediction')

# User input fields
age = st.number_input('Age:', min_value=0)
sex = st.selectbox('Sex:', ['male', 'female'])
bmi = st.number_input('BMI:', min_value=0.0)
children = st.number_input('Number of Children:', min_value=0)
smoker = st.selectbox('Smoker:', ['no', 'yes'])
region = st.selectbox('Region:', ['northeast', 'northwest', 'southeast', 'southwest'])

# Button to trigger prediction
if st.button('Predict Charges'):
    # Get prediction and display results  
    predicted_charges = predict_charges(model, age, sex, bmi, children, smoker, region)
    st.success(f"Predicted Insurance Charges: ${predicted_charges:.2f}")
