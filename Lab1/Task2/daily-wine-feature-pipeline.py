import os
import modal


LOCAL=True

#For deploying on modal
if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

#sets function to be executed every 24h on modal
   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

#Generates random wine data frame using CTGAN, using a GAN-based approach to create synthetic data based on the wine feature group
def get_random_wine(wine_fg):
    import pandas as pd
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    import random

    #Getting dataframe from feature group and getting its metadata
    wine_df = wine_fg.read()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=wine_df)

    #Synthesizing new data
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=True,
        epochs = 6
    )
    synthesizer.fit(wine_df)
    new_wines = synthesizer.sample(num_rows=100)

    #Randomly choosing one of the new synthesised wines and returning it
    random_index = random.randint(0, len(new_wines) - 1)
    random_wine = new_wines.iloc[random_index].to_frame().T
    print(random_wine)
    return random_wine


def g():
    import hopsworks
    import pandas as pd

    #Login to hopsworks and access wine feature group
    project = hopsworks.login()
    fs = project.get_feature_store()
    wine_fg = fs.get_feature_group(name="wine",version=3)

    #Gets new wine and inserts into feature group
    new_wine_df = get_random_wine(wine_fg)
    new_wine_df['wine_type'] = int(new_wine_df['wine_type'])
    new_wine_df['quality'] = int(new_wine_df['quality'])
    wine_fg.insert(new_wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
