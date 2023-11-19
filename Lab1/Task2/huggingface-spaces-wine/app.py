import streamlit as st
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=16)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type]], 
                      columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'wine_type'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    # prediction_url = "https://github.com/Lukox/KTH-ID2223/tree/main/Lab1/Task2/Assets/" + str(res[0]) + ".png"
    # img = Image.open(requests.get(prediction_url, stream=True).raw)            
    return res

# Streamlit app
st.title("Wine Quality Predictive Analytics")
st.write("Predicting wine quality given its fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, density, ph, sulphates, alcohol and wine type")

# Input fields
fixed_acidity = st.number_input("Fixed acidity", value=2.0)
volatile_acidity = st.number_input("Volative acidity", value=1.0)
citric_acid = st.number_input("Citric acid", value=2.0)
residual_sugar = st.number_input("Residual sugar", value=1.0)
chlorides = st.number_input("Chlorides", value=1.0)
free_sulfur_dioxide = st.number_input("Free sulfur dioxide", value=1.0)
density = st.number_input("Density", value=1.0)
ph = st.number_input("ph", value=1.0)
sulphates = st.number_input("Sulphates", value=1.0)
alcohol = st.number_input("Alcohol", value=1.0)
wine_type = st.number_input("Wine type", value=1.0)

# Button to trigger prediction
if st.button("Predict"):
    # Make prediction
    prediction = wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type)
    
    # Display prediction
    st.write("Prediction:", prediction)