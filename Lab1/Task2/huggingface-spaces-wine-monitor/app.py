import streamlit as st
from PIL import Image
import hopsworks

#login into hopsworks
project = hopsworks.login()
fs = project.get_feature_store()


dataset_api = project.get_dataset_api()

#download the images of the latest wine prediction (should be updated daily), stored in Resouces/images on hopsworks
dataset_api.download("Resources/images/latest_prediction.png" ,overwrite=True)
dataset_api.download("Resources/images/correct_prediction.png",overwrite=True)
dataset_api.download("Resources/images/df_recent.png",overwrite=True)
dataset_api.download("Resources/images/wine_confusion_matrix.png",overwrite=True)

#create a title and 2 columns to have the predicted quality and actual quality side by side
st.title("Prediction Dashboard")
col1, col2 = st.columns(2, gap="large")

#today's predicted image
with col1:
    st.subheader("Today's Predicted Image")
    predicted_img = st.image("latest_prediction.png", caption="Predicted Quality")

#today's actual image
with col2:
    st.subheader("Today's Actual Image")
    actual_img = st.image("correct_prediction.png", caption="Actual Quality")
    
#recent prediction history image
st.subheader("Recent Prediction History")
recent_predictions_img = st.image("df_recent.png", caption="Recent Predictions")

#confusion matrix
st.subheader("Confusion Matrix")
confusion_matrix_img = st.image("wine_confusion_matrix.png", caption="Confusion Matrix")
