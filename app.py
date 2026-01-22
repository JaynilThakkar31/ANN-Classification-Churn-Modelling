import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

#Load the trained model, scaler and OHE
model = tf.keras.models.load_model('model.h5')

# load the encoder and scaler
with open('One_Hot_encoding.pkl','rb') as file:
    Ohe_encoder = pickle.load(file)

with open('label_encoded_gen.pkl','rb') as file:
    label_encoded_gen = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title('Customer Churn Prediction')

#User Input

geography = st.selectbox('Geography',Ohe_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoded_gen.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0 ,10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#Prepare Input Data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoded_gen.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

#OHE 'Geography'
geo_encoded = Ohe_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=Ohe_encoder.get_feature_names_out(['Geography']))

# concat OHE and remove geography
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scaler the input data
input_scaler = scaler.transform(input_data)

#Prediction Churn
prediction = model.predict(input_scaler)
prediction_prob = prediction[0][0]

st.write(f'Chirn Probability : {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')