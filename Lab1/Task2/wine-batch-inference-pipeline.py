import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.2.2","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=16)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=2)
    batch_data = feature_view.get_batch_data()
    y_pred = model.predict(batch_data)
        
    offset = 1
    wine_quality = y_pred[y_pred.size-offset]
    print("Predicted Wine Quality: " + str(wine_quality))

    if int(wine_quality) % 2 == 0:
        os.makedirs("../resources/images/", exist_ok=True)
        with open("../resources/images/latest_prediction.jpg", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(wine_quality) +".jpg"
                ).content
            )
        dataset_api = project.get_dataset_api()    
        dataset_api.upload("../resources/images/latest_prediction.jpg", "Resources/images", overwrite=True)  
    else:
        os.makedirs("../resources/images/", exist_ok=True)
        with open("../resources/images/latest_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(wine_quality) +".gif"
                ).content
            )
        dataset_api = project.get_dataset_api()    
        dataset_api.upload("../resources/images/latest_prediction.gif", "Resources/images", overwrite=True)

   
    wine_fg = fs.get_feature_group(name="wine", version=2)
    df = wine_fg.read() 
    #print(df)
    label = df.iloc[-offset]["quality"]
    label = int(label)
    print("Actual Wine Quality: "+ str(label))


    if label % 2 == 0:
        with open("../resources/images/correct_prediction.jpg", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(label) +".jpg"
                ).content
            )

        dataset_api.upload(
            "../resources/images/correct_prediction.jpg", "Resources/images", overwrite=True
        )
    else:
        with open("../resources/images/correct_prediction.gif", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(label) +".gif"
                ).content
            )

        dataset_api.upload(
            "../resources/images/correct_prediction.gif", "Resources/images", overwrite=True
        )

    os.makedirs("../resources/images/", exist_ok=True)
    with open("../resources/images/latest_prediction.png", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(wine_quality) +".png"
                ).content
            )
    dataset_api.upload("../resources/images/latest_prediction.png", "Resources/images", overwrite=True)
    os.makedirs("../resources/images/", exist_ok=True)
    with open("../resources/images/correct_prediction.png", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(label) +".png"
                ).content
            )
    dataset_api.upload("../resources/images/correct_prediction.png", "Resources/images", overwrite=True)


    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
            'prediction': [wine_quality],
            'label': [label],
            'datetime': [now],
        }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(
        df_recent, "../resources/images/df_recent.png", table_conversion="matplotlib"
    )
    dataset_api.upload(
        "../resources/images/df_recent.png", "Resources/images", overwrite=True
    )
    
    #HISTORY
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 6:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True 3', 'True 4', 'True 5', 'True 6', 'True 7', 'True 8', 'True 9'],
                            ['Pred 3', 'Pred 4', 'Pred 5', 'Pred 6', 'Pred 7', 'Pred 8', 'Pred 9'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 6 different wine quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 6 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

