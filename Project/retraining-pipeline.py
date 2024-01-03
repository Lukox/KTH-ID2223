import os
import modal
import json
from io import BytesIO
import pickle
import pandas as pd


LOCAL=False
MODEL_DIR = "/models"

if LOCAL == False:
    
    volume = modal.NetworkFileSystem.persisted("league")
    stub = modal.Stub("league-training")
    image = modal.Image.debian_slim().pip_install("pandas","google-auth", "google-auth-oauthlib", "google-auth-httplib2", "google-api-python-client", "tensorflow", "scikit-learn", "keras==2.12.0")

    @stub.function(image=image, timeout = 2000, secrets=[modal.Secret.from_name("my-googlecloud-secret-2"), modal.Secret.from_name("RIOT_API_KEY"),], network_file_systems={MODEL_DIR: volume})
    def f():
        g()

def getFile(filename, drive_service):
    file_name = filename
    results = drive_service.files().list(q=f"name='{file_name}'", fields="files(id)").execute()
    files = results.get('files', [])
    return files

def getDataset(drive_service):
    files = getFile("NewData.csv", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pd.read_csv(content)
        return df
    else:
        print(f"File NewData.csv not found.")

def trainModel(data):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    X = data.drop("teamWin", axis=1)
    y = data["teamWin"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_params = {'activation': 'relu', 'dropout_rate': 0.4, 'hidden_units': [50], 'optimizer': 'rmsprop', 'weight_decay': 0.0001}

    # Create the model with the best parameters
    best_model = Sequential()
    for units in best_params['hidden_units']:
        best_model.add(Dense(units=units, activation=best_params['activation'], kernel_regularizer='l2'))
        best_model.add(Dropout(best_params['dropout_rate']))
    best_model.add(Dense(units=1, activation='sigmoid'))
    best_model.compile(loss='binary_crossentropy', optimizer=best_params['optimizer'], metrics=['accuracy'])

    # Train the model with the entire training set
    best_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred.round())
    print("Test Accuracy with Best Parameters: {:.2f}%".format(accuracy * 100))
    return best_model

def saveModel(drive_service):
    from googleapiclient.http import MediaFileUpload

    files = getFile("model.h5", drive_service)
    if files:
        print("saving")
        file_metadata = {
            "name": "model.h5",
        }
        file_id = files[0]['id']
        filepath = MODEL_DIR+"/model.h5"
        media = MediaFileUpload(filepath, mimetype='application/zip')
        drive_service.files().update(body=file_metadata,fileId=file_id, media_body = media).execute()

def g():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    from tensorflow import keras

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=credentials)
    data = getDataset(drive_service)
    model = trainModel(data)
    model.save(MODEL_DIR+"/model.h5")
    saveModel(drive_service)



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("league-training")
        with stub.run():
            f()