import streamlit as st
from PIL import Image
import requests
import pandas as pd
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
import json
import pickle
from googleapiclient.http import MediaIoBaseDownload
import tempfile

#model = keras.models.load_model("/content/drive/MyDrive/ML/model.h5")

def getFile(filename, drive_service):
    file_name = filename
    results = drive_service.files().list(q=f"name='{file_name}'", fields="files(id)").execute()
    files = results.get('files', [])
    return files

def getDataset(drive_service):
    files = getFile("data.csv", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pd.read_csv(content)
        return df
    else:
        print(f"File data.csv not found.")

def getModel(drive_service):
    files = getFile("model.h5", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        model_stream = BytesIO()
        downloader = MediaIoBaseDownload(model_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp.write(model_stream.getvalue())
        temp.close()

        # Load the model using Keras
        model = keras.models.load_model(temp.name)
        return model
    else:
        print(f"Model not found.")
    
def getSynergyMatrix(drive_service):
    files = getFile("NewSynergyMatrix.pkl", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pickle.load(content)
        return df
    else:
        print(f"File synergyMatrix.pkl not found.")

def getCounterMatrix(drive_service):
    files = getFile("NewCounterMatrix.pkl", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pickle.load(content)
        return df
    else:
        print(f"File synergyMatrix.pkl not found.")

def getWinMatrix(drive_service):
    files = getFile("NewWinRateMatrix.pkl", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pickle.load(content)
        return df
    else:
        print(f"File winRateMatrix.pkl not found.")


print("START")

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import service_account

service_account_info = json.loads(st.secrets["credentials"])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info,
    scopes=['https://www.googleapis.com/auth/drive']
)
drive_service = build('drive', 'v3', credentials=credentials)

synergyMatrix = getSynergyMatrix(drive_service)
counterMatrix = getCounterMatrix(drive_service)
champion_stats = getWinMatrix(drive_service)
model = getModel(drive_service)
print("DONE")