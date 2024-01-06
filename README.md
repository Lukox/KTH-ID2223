# KTH-ID2223 : Scalable Machine Learning and Deep Learning Course

# Project: Pre-game League of Legends Predictor

## Introduction

League of Legends is a highly popular multiplayer online battle arena (MOBA) game developed by Riot Games. Released in 2009, it features two teams of five players each, choosing unique champions with distinct abilities to compete in strategic battles. The goal is to destroy the opposing team's Nexus in their base. Known for its dynamic gameplay and diverse champion roster, League of Legends attracts millions of players and viewers worldwide. 

Before every game, teams strategically select their champions, shaping the battlefield and influencing the outcome. This is known as drafting. The process involves alternating picks and bans, with each team aiming to create a well-rounded and synergistic composition while countering the opponent's choices. Champions are chosen based on their roles, strengths, and potential interactions with teammates. A successful draft can give a team a strategic advantage, setting the stage for intense and dynamic gameplay. We propose a model to predict the outcome of a League of Legends game right after the drafting stage, beforne any game actions.

## The Dataset

Utilizing the Riot API, we are actively gathering comprehensive data for our machine learning model aimed at predicting pre-game outcomes in League of Legends. This API provides real-time information on player statistics, match histories, and in-game dynamics, allowing us to identify patterns and key factors influencing match results. By leveraging this data, our model is designed to offer accurate and dynamic predictions, adapting to the evolving nuances of League of Legends gameplay.

We used various API endpoints to collect the data we need. First we used the League-V4 endpoint to find random players in all ranks of League of Legends from which we obtained 100,000 most recent matches. We filtered the games to be ranked, which is a competitive gamemode in order to reduce unpredictability and ensure maximum accuracy. From each of the games, we gathered data, including champions, positions, and more, from which we made our dataset. 

![Riot API](https://codepull.com/wp-content/uploads/2022/06/image-1-1024x406.png)

We produced models on two different datasets. The first one being raw data, which was just the champions drafted in each team using one hot encoding. The second dataset included engineered features, displayed and explained below: 

![Features](https://github.com/Lukox/KTH-ID2223/blob/main/Project/assets/features.png)

- **`blueTopRating`**: This metric represents the win rating of the blue side's top lane, ranging from -1 to 1, where 1 indicates a higher likelihood of winning.
- **`blueJgRating`**: Similar to `blueTopRating`, this parameter denotes the win rating for the blue side's jungle role.
- **`blueMidRating`**: Reflecting the win rating for the blue side's mid lane, this score helps assess the team's performance in the middle lane.
- **`blueBotRating`**: Corresponding to the blue side's bottom lane (AD Carry), this metric provides a win rating for that specific role.
- **`blueSupRating`**: Specifically for the blue side's support role, this rating indicates the likelihood of winning based on historical data.
- **`redTopRating`**, **`redJgRating`**, **`redMidRating`**, **`redBotRating`**, **`redSupRating`**: Analogous to their blue counterparts, these metrics provide win ratings for the respective roles on the red side.
- **`blueAvgMasteryLevel`**: This parameter signifies the average mastery level of champions for the blue team, offering insights into their collective champion proficiency.
- **`blueAvgMasteryPoints`**: Similar to `blueAvgMasteryLevel`, this metric represents the average mastery points of champions on the blue team, providing a measure of their experience with their chosen champions.
- **`redAvgMasteryLevel`**, **`redAvgMasteryPoints`**: These are the red team equivalents of the `blueAvgMasteryLevel` and `blueAvgMasteryPoints`, representing the average mastery level and points for the red team.
- **`blueSynergyScore`**, **`redSynergyScore`**: These scores range from -1 to 1, indicating how well the champions in each team work together. It is calculated based on the frequency of winning games when specific champion combinations are present on the same team.
- **`topDelta`**, **`jgDelta`**, **`midDelta`**, **`botDelta`**, **`supDelta`**: These parameters reflect the frequency with which champions in each role on the blue team beat their counterparts on the red team. A value of -1 indicates a higher frequency for the blue team, while 1 indicates a higher frequency for the red team.
- **`teamWin`**: This binary parameter indicates the overall outcome of the game. A value of 0 denotes a victory for the blue team, while a value of 1 signifies a victory for the red team.

The ratings for the Synergies, Deltas and Winrates were taken from data of 100,000 games, which we stores in matrices, providing information on champion win and losses, how often champions win together and how often they lose against others. 

Our data turned out to be relatively balanced, with 5015 wins on blue side and 4985 wins on red side out of 10000 games, making it a suitable dataset for our prediction model. 

## The Method

In our first model creation approach, we started with raw data, specifically the champion IDs of each player. Employing feature crosses, we generated additional features and labeled the team indicator. Our primary model for this approach was a neural network. We split the data into training and testing sets, processed the raw features, and trained the neural network to predict game outcomes. For the second approach, we utilized engineered features derived from extensive data analysis. Our modeling efforts involved experimenting with various machine learning models, including a Random Forest Classifier and different neural network architectures. To optimize the neural network's hyperparameters, we employed black-box optimization techniques such as grid search. The objective was to enhance the model's predictive performance by systematically exploring the hyperparameter space. Both approaches aimed to predict the pre-game outcomes of League of Legends matches, with the first focusing on raw data and feature crosses, and the second leveraging engineered features and advanced model selection techniques. The features were collected in our `backfill-feature-pipeline.ipynb`, and the training for both approaches was does in the `training-pipeline.ipynb`.

Since League of Legends has millions of games played every day, the dataset is very dynamic, thus everyday using Modal, we collect new matches and their data and combine it in our feature store on Google Drive with our historical data using the `feature-pipeline.py`. Every two weeks, whenever a new patch is released, we retrain the models on our new data with the `retraining-pipeline.py`, also using Modal on cloud so it is done automatically. The model is then saved in our feature store which we then download for our inference application. 
 
## The Results

In the results section, our initial raw data approach yielded a maximum accuracy of 54%, marginally surpassing random chance. This outcome suggests that predicting League of Legends game outcomes based solely on champion IDs and feature crosses might be challenging, indicating that draft dynamics alone may not be highly indicative of match results. It appears that the intrinsic variability introduced by player skill and decision-making may play a significant role in the unpredictability of game outcomes during the drafting phase.

In contrast, the engineered features approach demonstrated more promising results, achieving a maximum accuracy of 70.8%. This unexpected success, given the inherent unpredictability of League of Legends, underscores the significance of the engineered features in capturing crucial aspects of player performance. The higher accuracy suggests that our model, trained on these enhanced features, is better able to discern patterns and relationships that contribute to predicting game outcomes more effectively. This result prompts further exploration into the engineered features and the potential insights they offer into the intricate dynamics of League of Legends matches, emphasizing the importance of considering player performance as a crucial factor in pre-game predictions. 

We used our model in ou inference application on [HuggingFace Spaces](https://huggingface.co/spaces/Lukox/League) where one can input a summoner username and the application will display predictions and actual outcomes of the last few games. In the scenario the summoner is currently playing a game, an extra feature is the live prediction of the outcome of that game. 

## Limitations 
One major limitation of our study lies in the constraints of data collection through the Riot API. The API imposes limitations on the number of calls per second and minute, hindering our ability to gather a more extensive dataset efficiently. For instance, the retrieval of data for approximately 10,000 games required around 200,000 API calls, spanning a time frame of approximately 11 days. Improved access to the Riot API, with a higher rate limit, could significantly improve data collection and enable a more comprehensive exploration of League of Legends match data.

Another limitation our study was the RIOT API was often unvailable, which halted progress as data could no longer be collected.

## Further Explorations
To enhance the robustness of our predictive models, further exploration could take multiple approaches. Firstly, focusing exclusively on the highest ranks or specific elos may offer insights into games where player performance is more consistent and there is less inherent unpredictability. This targeted approach could potentially yield higher accuracy scores and more reliable predictions.

Additionally, incorporating individual player performance metrics, such as win rates, winning or losing streaks, and other relevant statistics, could be a interesting area for exploration. These player-centric features might provide valuable context, allowing the model to better account for the influence of individual player skills and tendencies on game outcomes. However, it's important to note that implementing this approach effectively would require a substantially higher number of API calls.

In conclusion, overcoming the limitations posed by API constraints and refining the focus of our dataset to include higher-ranked games or individual player performance metrics could unlock new dimensions of understanding and significantly improve the accuracy of our predictive models in forecasting League of Legends pre-game outcomes.

## How to run
We ran the files in this order:
1) `backfill-feature-pipeline.ipynb` - collects the matches and transforms raw data into features, then stores it on Google Drive.
2) `training-pipeline.ipynb` - trains the model and uploads it on Google Drive.
3) `feature-pipeline.py` - collects new matches every day and combine them to previous data on Google Drive. 
4) `retraining-pipeline.py` - retrains the model on the updated dataset every 2 weeks and uploads it to Google Drive.
5) `app.py` - HuggingFace Spaces UI where player can input their summoner name to obtain the game predictions of their live game or their last couple of games.

   
![UI](https://github.com/Lukox/KTH-ID2223/blob/main/Project/assets/ui.png)

# Lab 2

## Fine-tuning Whisper

For Lab 2, we fine-tuned a transformer for language transcription to our mother tongue: polish. The pre-trained transformer model we used was [Whisper-small](https://huggingface.co/openai/whisper-small). We used the [Common_Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) polish dataset for our data, and prepared it for the whisper model in a shared Google Colab - [Feature_Pipeline.ipynb](https://github.com/Lukox/KTH-ID2223/blob/main/Lab2/Feature_Pipeline.ipynb). We then saved the prepared data, which was split into train and test data, inside Google Drive, which acted as our feature store. To fine tune the whisper-small model, we created a [training pipeline](https://github.com/Lukox/KTH-ID2223/blob/main/Lab2/Training_Pipeline.ipynb), which loaded the data from the drive and performed the training. We used the Google Colab Free Tier GPU - T4, which was only given to us for a limited amount of time, thus we had to save checkpoints and resume training from them once our GPU was available again. In the end we achieved a WER of 25.9 with this dataset and whisper-small pretrained model.

## Interface

We implemented a simple graphical user interface on Huggingface with Streamlit. One would be able to input a Youtube link of any video, which would then be transcribed by our model. Then, one can can choose a language to translate the transcribed text to, or keep the transcribed text in polish. Later, the original video is available to be played with the new transcribed text as subtitles corresponding to the sound of the video. The UI is avaliable on [Hugginface](https://huggingface.co/spaces/Lukox/Whisper)

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
