import streamlit as st
from PIL import Image
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()

dataset_api.download("Resources/images/latest_prediction.png" ,overwrite=True)
dataset_api.download("Resources/images/correct_prediction.png",overwrite=True)
dataset_api.download("Resources/images/df_recent.png",overwrite=True)
dataset_api.download("Resources/images/confusion_matrix.png",overwrite=True)


st.title("Prediction Dashboard")
col1, col2 = st.columns(2)

#Today's Predicted Image and Actual Image
with col1:
    st.subheader("Today's Predicted Image")
    predicted_img = st.image("latest_prediction.png", caption="Predicted Quality")

#Recent Prediction History and Confusion Matrix
    st.subheader("Recent Prediction History")
    recent_predictions_img = st.image("df_recent.png", caption="Recent Predictions")

with col2:
    st.subheader("Today's Actual Image")
    actual_img = st.image("correct_prediction.png", caption="Actual Quality")

    st.subheader("Confusion Matrix with Historical Prediction Performance")
    confusion_matrix_img = st.image("confusion_matrix.png", caption="Confusion Matrix")
