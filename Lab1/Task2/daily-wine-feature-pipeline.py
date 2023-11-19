import os
import modal


LOCAL=True

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_wine():
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "fixed_acidity": [random.uniform(3.8, 15.9)],
                       "volatile_acidity": [random.uniform(0.08, 1.58)],
                       "citric_acid": [random.uniform(0, 1.66)],
                       "residual_sugar": [random.uniform(0.6, 65.8)],
                       "chlorides": [random.uniform(0.009, 0.611)],
                       "free_sulfur_dioxide": [random.uniform(1, 289)],
                       "density": [random.uniform(0.98711, 1.03898)],
                       "ph": [random.uniform(2.72, 4.01)],
                       "sulphates": [random.uniform(0.22, 2)],
                       "alcohol": [random.uniform(8, 14.9)],
                       "wine_type": [int(random.uniform(0, 1))],
                       "quality": [int(random.uniform(3, 9))]
                      })
    return df


def get_random_wine(wine_fg):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    import random

    wine_df = wine_fg.read()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=wine_df)

    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=True,
        epochs = 6
    )
    synthesizer.fit(wine_df)
    new_wines = synthesizer.sample(num_rows=100)
    #print(new_wines)


    random_index = random.randint(0, len(new_wines) - 1)
    random_wine = new_wines.iloc[random_index].to_frame().T
    print(random_wine)
    return random_wine


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    wine_fg = fs.get_feature_group(name="wine",version=2)
    #get_random_wine(wine_fg)
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
