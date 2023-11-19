import streamlit as st
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd
from io import BytesIO

#login into hopsworks project (API keys is a secret in huggingface which allows the correct project to be opened)
project = hopsworks.login()
fs = project.get_feature_store()

#download the training model from hopsworks 
mr = project.get_model_registry()
model = mr.get_model("wine_model", version=16)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

#function which takes the parameters (the features) and uses the training model to predict the quality of a wine with the given parameters
def predict_wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type):
 
    #create a dataframe with the inputted parameters
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type]], 
                      columns=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'wine_type'])
  
    print(df)
    
    #predict using the model: result = the predicted quality of the wine
    result = model.predict(df)

    st.subheader("Predicted quality of the wine: " + str(result[0]))
    print(result)
    
    #even values for quality outputs a jpg, while odd values for quality outputs a gif (for amusing visual purposes)
    if (result[0] % 2 == 0):
        print("even")
        prediction_url = "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/" + str(result[0]) + ".jpg"
        img = Image.open(BytesIO(requests.get(prediction_url).content))
        st.image(img)
    else:
        print("odd")
        prediction_url = "![Alt Text](https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/" + str(result[0]) + ".gif)"
        st.markdown(prediction_url)

#using Streamlit ouput the title and heading
st.title("Wine Quality Predictive Analytics")
st.write("Predicting wine quality given its fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, density, ph, sulphates, alcohol and wine type")

# create input fields - variables change as soon as the value in the input box changes
fixed_acidity = st.number_input("Fixed acidity", value=None, format="%f")
volatile_acidity = st.number_input("Volative acidity", value=None, format="%f")
citric_acid = st.number_input("Citric acid", value=None, format="%f")
residual_sugar = st.number_input("Residual sugar", value=None, format="%f")
chlorides = st.number_input("Chlorides", value=None, format="%f")
free_sulfur_dioxide = st.number_input("Free sulfur dioxide", value=None, format="%f")
density = st.number_input("Density", value=None, format="%f")
ph = st.number_input("ph", value=None, format="%f")
sulphates = st.number_input("Sulphates", value=None, format="%f")
alcohol = st.number_input("Alcohol", value=None, format="%f")
wine_type = st.selectbox("Wine type", ["Red", "White"])

# Button to trigger prediction by calling the wine() function
if st.button("Predict"):
    
    #convert Red or White into numerical values, as red is saved as 0 and white is saved as 1
    if (wine_type == "Red"):
        wine_type = 0
    else:
        wine_type = 1

    # Make prediction
    predict_wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, density, ph, sulphates, alcohol, wine_type)
    