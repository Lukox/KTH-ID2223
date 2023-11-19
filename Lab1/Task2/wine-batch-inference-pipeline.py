import os
import modal
    
LOCAL=True

#install the required dependencies on Modal and make sure it is ran once a day
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

    #login into hopsworks 
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    #download the training model from hopsworks 
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=17)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    #get the batch data (a large set of data) from the feature view
    feature_view = fs.get_feature_view(name="wine", version=4)
    batch_data = feature_view.get_batch_data()

    #generate a set of predictions from the batch data
    y_pred = model.predict(batch_data)

    #offset represents the index of the data, with offset = 1 being the last row of data, offset = 2 being the second last row of data and so on    
    offset = 1

    #outputs the predicted quality given the offset. The code segment below takes the new wine generated daily by the wine generator function 
    # (which will be the last row of data, hence offset = 1) and creates a predicted quality with the given feature values. Using the quality value,
    # images for the predicted quality, actual quality, recent prediction history and confusion matrix are produced and saved in hopsworks
    # so that the wine-monitor app can access and display it on huggingface UI.

    #wine_quality = predicted quality
    wine_quality = round(y_pred[y_pred.size-offset])
    print("Predicted Wine Quality: " + str(wine_quality))

   # the actual value for quality of the newly generated wine is taken from the feature group, as the newly generated wine is stored there 
    wine_fg = fs.get_feature_group(name="wine", version=3)
    df = wine_fg.read() 

    #label = actual quality
    label = df.iloc[-offset]["quality"]
    label = int(label)
    print("Actual Wine Quality: "+ str(label))

    dataset_api = project.get_dataset_api()
        
    #if directory doesn't exist, create one to store the predicted quality and actual quality images
    os.makedirs("../resources/images/", exist_ok=True)
    with open("../resources/images/latest_prediction.png", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(wine_quality) +".png"
                ).content
            )
    #upload the predicted quality image on hopsworks
    dataset_api.upload("../resources/images/latest_prediction.png", "Resources/images", overwrite=True)
    
    os.makedirs("../resources/images/", exist_ok=True)
    with open("../resources/images/correct_prediction.png", "wb") as gif:
            gif.write(
                requests.get(
                    "https://raw.githubusercontent.com/Lukox/KTH-ID2223/main/Lab1/Task2/Assets/"+ str(label) +".png"
                ).content
            )
    #upload the actual quality image on hopsworks
    dataset_api.upload("../resources/images/correct_prediction.png", "Resources/images", overwrite=True)

    #get the feature group to create a recent prediction history table
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine Quality Prediction/Outcome Monitoring"
                                                )
    
    #get the time string representation of date and time of the new prediction and create a dataframe with predicted quality, actual quality and datetime
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
            'prediction': [wine_quality],
            'label': [label],
            'datetime': [now],
        }
    #monitor_df is the dataframe of the newly generated wine
    monitor_df = pd.DataFrame(data)

    #inserts the new prediction into wine_predictions feature group
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    #history_df contains all the data of wines from the feature group
    history_df = monitor_fg.read()

    #combine the dataframe of the new wine with the rest of the wines
    history_df = pd.concat([history_df, monitor_df])

    #create a table with the 4 latest predictions and upload it onto hopsworks
    df_recent = history_df.tail(4)
    dfi.export(
        df_recent, "../resources/images/wine_df_recent.png", table_conversion="matplotlib"
    )
    dataset_api.upload(
        "../resources/images/wine_df_recent.png", "Resources/images", overwrite=True
    )
    
    #history of predictions and labels
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 7 wine qualities
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    #if predictions.value_counts().count() == 7:
    results = confusion_matrix(labels, predictions)

    df_cm = pd.DataFrame(results, ['True 3', 'True 4', 'True 5', 'True 6', 'True 7', 'True 8', 'True 9'],
                        ['Pred 3', 'Pred 4', 'Pred 5', 'Pred 6', 'Pred 7', 'Pred 8', 'Pred 9'])

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./wine_confusion_matrix.png")
    dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
    # else:
    #     print("You need 7 different wine quality predictions to create the confusion matrix.")
    #     print("Run the batch inference pipeline more times until you get 7 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()