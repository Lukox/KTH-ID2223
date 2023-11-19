# KTH-ID2223 : Scalable Machine Learning and Deep Learning Course - Lab 1

# Task 1

For Task 1, we had to build and deploy a serverless ML prediction system for the Iris Flower Dataset using Hopsworks, Modal and Hugging Face Spaces. The system should be able create a model which can predict the variety of an iris flower given the length and width of its petal and sepal, and then display it with a Gradio UI on Hugging Face Spaces. A new synthetically generated iris flower is also created once a day through the use of Modal and a predicted variety can be generated, as well as the actual variety, a table displaying the recent prediction histories and a confusion matrix. All the code was provided as part of the lab, but Hopsworks, Modal and Hugging Face had to be configured by us to connect the pipelines.

Hugging Face UIs:

Iris (interactive):
https://huggingface.co/spaces/Lukox/Iris

Iris monitor (daily flower predition dashboard):
https://huggingface.co/spaces/Lukox/Iris-monitor

# Task 2
For Task 2, we had to build and deploy a serverless ML prediction system for the Wine Quality Dataset, which contains the information about the various attributes of red and white wines, such as its density, pH, fixed acidity or amount of chlorides present, and the associated quality ratings. Using this dataset, we had to use feature engineering to drop attributes with little to no predictive power, and then create a training model which can be used to predict wine qualities given its attributes. 

Similarly to Task 1, we had to also create a wine generator function which generates a synthetic wine once a day, for which we run a prediction for its quality using our pre-trained model. To generate the synthetic wine, we used CTGAN, which uses a GAN-based approach to create synthetic data based on the wine feature group. This new data is then added to the feature group in hopsworks, and afterwards the batch inference pipeline is executed which generates a predicted quality for the new wine. This value and other statistics are then displayed on Hugging Face Space via a Streamlit UI.

We used a RandomForestRegression to train our data, which gave us the lowest value mean squared error value of 0.456 in comparison to other models. This model will predict a non-integer value for quality, which then will be rounded using Gaussian rounding (round()) to reduce rounding bias and to create an integer value for quality so it can be compared with the actual quality.

Hugging Face UIs:

Wine (interactive):
https://huggingface.co/spaces/Lukox/wine

Wine monitor (daily wine predition dashboard):
https://huggingface.co/spaces/Lukox/wine_monitor