# KTH-ID2223 : Scalable Machine Learning and Deep Learning Course

# Lab 2

## Fine-tuning Whisper

For Lab 2, we fine-tuned a transformer for language transcription to our mother tongue: polish. The pre-trained transformer model we used was [Whisper-small](https://huggingface.co/openai/whisper-small). We used the [Common_Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) polish dataset for our data, and prepared it for the whisper model in a shared Google Colab - [Feature_Pipeline.ipynb](https://github.com/Lukox/KTH-ID2223/blob/main/Lab2/Feature_Pipeline.ipynb). We then saved the prepared data, which was split into train and test data, inside Google Drive, which acted as our feature store. To fine tune the whisper-small model, we created a [training pipeline](https://github.com/Lukox/KTH-ID2223/blob/main/Lab2/Training_Pipeline.ipynb), which loaded the data from the drive and performed the training. We used the Google Colab Free Tier GPU - T4, which was only given to us for a limited amount of time, thus we had to save checkpoints and resume training from them once our GPU was available again. In the end we achieved a WER of 25.9 with this dataset and whisper-small pretrained model.

## Interface

We implemented a simple graphical user interface on Huggingface with Streamlit. One would be able to use a Youtube of any video, which would then be transcribed by our model. TO BE FINISHED

## Limitations

We faced challenges when trying to fine tune the whisper-small model. First of all, the GPU RAM from Colab was not large enough, so we had to reduce our batch size for our model among other things. Another way could have been using the whisper-base pre-trained model. Another problem was storage; the polish dataset was rather large and even on a paid Google Drive account we were lacking storage space, thus we had to cut down some of the training and test data. Due to a limited amount of data, there was a risk of overfitting, since the whisper model was trained on a much larger dataset. This could result in gradient exploding, as was seen in our experience where the WER went up eventually after a large amount of training steps.

## Improving Model Performance

There are two approaches to improving the model: model-centric and data-centric.

### Model Centric

For the **model-centric approach**, one can tune the hyperparameters. 
  - by increasing max_steps. This would train the model for a longer time. For our dataset, we used max_steps = 4000; however if we increased it, the WER could have decreased. A higher max_steps would essentially lead to more epochs, where with our implementation it was equivalent to 3 epochs. The problem with this is time limitation, requiring more Colab GPU usage.
  - Changing the optimizer. By default, the transformers trainer uses the AdamW optimizer. Modifying it can change the attributes of the neural network such as weights and learning rate in order to reduce the losses. 
  - changing the learning rate - higher learning rate may cause quicker optimization, but if too large we may miss the optimal solution.
  - changing the batch_size and gradient_accumulation_steps, which are strongly connected parameters. We were limited to a batch size of 8, as anything higher would produce the CUDA out of memory issue, essentially meaning there was not enough GPU RAM. A lower batch size can be more accurate, but can lead to high variance in gradients. A batch size too large can have low accuracy. Thus optimzing these parameters is important.

Other than tuning hyperparameters, we could have used a larger Whisper model, either whisper-medium or whisper-large. We attempted this, but again run into the same issue of GPU RAM. Upon research, one would require around 36GB of GPU RAM to fine tune the whisper-medium model, which is much higher than the 16GB provided by the T4 GPU. Another approach is using a different model, such as the [wav2vec2](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-polish). 

### Data Centric

For the **data-centric approach**, the training pipeline remains the same, and the focus is shifted to the dataset. Using a different larger dataset could lead to higher accuracy, or a lower WER, as the model would have more data to train on. This would potentially reduce overfitting issues that could have occured. To try out the data centric approach, we fine tuned our model with the same hyperparameters on the following datasets: 
  - [google/fleurs](https://huggingface.co/datasets/google/fleurs): This dataset is created by google and had audios and transcriptions for polish, as well as many other languages. The issue with this dataset was that it was really small. Hence, our training was much faster, but our WER was worse than the Common Voice dataset, only reaching a WER of 32.
  - [fsicoli/common_voice_15](https://huggingface.co/datasets/fsicoli/common_voice_15_0): This was another common voice dataset from the Mozilla Common Voice Corpus 15 project, but was different and had more data. However, to our surprise, we only managed a 38 WER.

We also attempted to train the model on combined datasets, but we either ran into disk storage issues or GPU RAM issues, or it did not improve the WER. Thus in the end, with our attempts at the data-centric approach, we could not improve our model, but we believe if we had more resources, we definitely could have improved it.

# Lab 1

## Task 1

For Task 1, we had to build and deploy a serverless ML prediction system for the Iris Flower Dataset using Hopsworks, Modal and Hugging Face Spaces. The system should be able create a model which can predict the variety of an iris flower given the length and width of its petal and sepal, and then display it with a Gradio UI on Hugging Face Spaces. A new synthetically generated iris flower is also created once a day through the use of Modal and a predicted variety can be generated, as well as the actual variety, a table displaying the recent prediction histories and a confusion matrix. All the code was provided as part of the lab, but Hopsworks, Modal and Hugging Face had to be configured by us to connect the pipelines.

Hugging Face UIs:

Iris (interactive):
https://huggingface.co/spaces/Lukox/Iris

Iris monitor (daily flower predition dashboard):
https://huggingface.co/spaces/Lukox/Iris-monitor

## Task 2
For Task 2, we had to build and deploy a serverless ML prediction system for the Wine Quality Dataset, which contains the information about the various attributes of red and white wines, such as its density, pH, fixed acidity or amount of chlorides present, and the associated quality ratings. Using this dataset, we had to use feature engineering to drop attributes with little to no predictive power, and then create a training model which can be used to predict wine qualities given its attributes. 

Similarly to Task 1, we had to also create a wine generator function which generates a synthetic wine once a day, for which we run a prediction for its quality using our pre-trained model. To generate the synthetic wine, we used CTGAN, which uses a GAN-based approach to create synthetic data based on the wine feature group. This new data is then added to the feature group in hopsworks, and afterwards the batch inference pipeline is executed which generates a predicted quality for the new wine. This value and other statistics are then displayed on Hugging Face Space via a Streamlit UI.

We used a RandomForestRegression to train our data, which gave us the lowest value mean squared error value of 0.456 in comparison to other models. This model will predict a non-integer value for quality, which then will be rounded using Gaussian rounding (round()) to reduce rounding bias and to create an integer value for quality so it can be compared with the actual quality.

Hugging Face UIs:

Wine (interactive):
https://huggingface.co/spaces/Lukox/wine

Wine monitor (daily wine predition dashboard):
https://huggingface.co/spaces/Lukox/wine_monitor
