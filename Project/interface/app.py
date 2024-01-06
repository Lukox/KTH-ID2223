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
from sklearn.preprocessing import StandardScaler
import tempfile
import time

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en,pl-PL;q=0.9,pl;q=0.8,en-US;q=0.7",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": st.secrets["RIOT_API"]
    }

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

#get mastery given championId and a player's puuid, returns both level and points
def getMastery(championId, puuid):
  url = f"https://euw1.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{championId}"
  r = requests.get(url=url, headers=headers)
  if(r.status_code == 200):
    r = r.json()
    return  r["championLevel"], r["championPoints"]
  else:
    print("Error getting mastery")
    return 0, 0

#get champion data given match data in json format
def getChampions(r):
    champions = [[participant["championName"],participant["championId"],participant["teamPosition"], participant["win"]] for participant in r["info"]["participants"]]
    return champions

#get all participants in a game given match data in json format
def getMatchParticipants(r):
  participants = [[participant["summonerId"], participant["puuid"], participant["summonerName"]] for participant in r["info"]["participants"]]
  return participants

#get the average rating for an out of position champion
def getAutofillRating(champion_stats):
    totalWins = 0
    totalLosses = 0

    for champId, positions in champion_stats.items():
        for position, stats in positions.items():
            if stats['wins'] + stats['losses'] < 50:
                totalWins += stats['wins']
                totalLosses += stats['losses']
    avgAutofillRating = (totalWins - totalLosses)/ (totalWins + totalLosses)
    return avgAutofillRating

#determine which team won the game
def getMatchWinner(matchData):#, index):
  if(matchData[0]["win"] == True):
    return 0
  else:
    return 1

#determine a champions rating for given position
def getWinLoss(champion_id, position_index, avgAutofillRating, champion_stats):
    position_mapping = {0: 'TOP', 1: 'JUNGLE', 2: 'MIDDLE', 3: 'BOTTOM', 4: 'UTILITY'}

    if position_index not in position_mapping:
        raise ValueError("Invalid position index")

    position = position_mapping[position_index]

    # Get wins and losses for the specified champion and position
    stats = champion_stats[champion_id].get(position, {'wins': 0, 'losses': 0})
    wins = stats['wins']
    losses = stats['losses']

    if(wins+losses < 50):
      return avgAutofillRating
    else:
      return (wins - losses)/(wins + losses)

#determine team synergies
def getSynergy(matchData, synergyMatrix, keyToIndex):
    blueRating = 0
    redRating = 0
    for i in range(5):
      championKey = matchData[i]["championId"]
      for j in range(i+1, 5):
        teammateKey = matchData[j]["championId"]
        wins = synergyMatrix[keyToIndex[championKey]][keyToIndex[teammateKey]]["wins"]
        losses = synergyMatrix[keyToIndex[championKey]][keyToIndex[teammateKey]]["losses"]
        blueRating += ((wins - losses)/(wins + losses))

    for i in range(5, 10):
      championKey = matchData[i]["championId"]
      for j in range(i+1, 10):
        teammateKey = matchData[j]["championId"]
        wins = synergyMatrix[keyToIndex[championKey]][keyToIndex[teammateKey]]["wins"]
        losses = synergyMatrix[keyToIndex[championKey]][keyToIndex[teammateKey]]["losses"]
        redRating += ((wins - losses)/(wins + losses))

    return blueRating, redRating

#determine counter lane matchups 
def getDeltas(matchData, counterMatrix, keyToIndex):
  deltas = []

  for i in range(5):
      enemyKey = matchData[i+5]["championId"]
      championKey = matchData[i]["championId"]
      wins = counterMatrix[keyToIndex[championKey]][keyToIndex[enemyKey]]["wins"]
      losses = counterMatrix[keyToIndex[championKey]][keyToIndex[enemyKey]]["losses"]
      if wins+losses == 0:
        deltas.append(0)
      else:
        deltas.append((wins-losses)/(wins+losses))
  return deltas[0], deltas[1], deltas[2], deltas[3], deltas[4]

def getPuuid(name, headers):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en,pl-PL;q=0.9,pl;q=0.8,en-US;q=0.7",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": st.secrets["RIOT_API"]
    }
    print("requesting puuid")
    url = f"https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{name}"
    r = requests.get(url=url, headers=headers)
    if(r.status_code == 200):
        return r.json()["puuid"]
    else:
        print("No summoner by such name in this region")

def getMatchId(name, headers, i):
    puuid = getPuuid(name, headers)
    time.sleep(1)
    print("requesting match")
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start={i}&count=1"
    r = requests.get(url=url, headers=headers)
    if(r.status_code == 200):
        return r.json()[0], puuid
    else:
        print("Could not load ranked games for summoner")

def style_columns(val, index_val):
    if (teamWin == 0):
       colour = ['lightblue','red']
    else:
       colour = ['red','lightblue']

    styles = f'background-color: {colour[index_val]}'

    return styles


def live_game_status(player_name, headers):
    url4 = f"https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{player_name}"
    r = requests.get(url=url4, headers=headers)
    time.sleep(1)
    if(r.status_code == 200):
        encrypted_summonerID = r.json()["id"]
        url4 = f"https://euw1.api.riotgames.com/lol/spectator/v4/active-games/by-summoner/{encrypted_summonerID}"
        r = requests.get(url=url4, headers=headers)
        time.sleep(1)
        if(r.status_code == 200):
           return True , r , encrypted_summonerID              
        else:
           print("Player currently is not in game")
           return False , r , encrypted_summonerID  
    else:
       print("No summoner by such name")
       
def live_game_data(r):
    participant = r.json()["participants"]

    url = "https://ddragon.leagueoflegends.com/cdn/13.24.1/data/en_US/champion.json"
    r_champ = requests.get(url=url)
    r_champ = r_champ.json()
    champions = []
    summoners = []
    for i in range (10):
        championId = participant[i]["championId"]
        summonerName = participant[i]["summonerName"]
        summonerId = participant[i]["summonerId"]
        puuid = participant[i]["puuid"]
        champion_name = None
        #puuid = getPuuid(summonerName, 'euw', headers)
        for champion, champion_data in r_champ['data'].items():
            if int(champion_data["key"]) == championId:
                champion_name = champion
                break
        summoners.append([summonerId, puuid, summonerName])
        champions.append([champion_name, championId])

    return summoners, champions  

def champion_positions(r):
    position_mapping = {0: 'TOP', 1: 'JUNGLE', 2: 'MIDDLE', 3: 'BOTTOM', 4: 'UTILITY'}
    order = [None]*10
    dups = []
    for i in range (10):
        max = 0
        champion_id = r.json()["participants"][i]["championId"]
        for x in range (5):
            if x not in position_mapping:
                raise ValueError("Invalid position index")

            position = position_mapping[x]

        # Get wins and losses for the specified champion and position
            stats = champion_stats[champion_id].get(position, {'wins': 0, 'losses': 0})
            wins = stats['wins']
            losses = stats['losses']
            if (wins+losses > max):
                main_role = x
                max = wins+losses

        

        if (i < 5):
            if (order[main_role] != None):
                dups.append(i)
            else:            
                order[main_role] = i
        else:
            if (order[main_role+5] != None):
                dups.append(i)
            else: 
                order[main_role+5] = i
    index = 0        
    for b in range(10):
        if (order[b] == None):
            order[b] = dups[index]
            index+=1
     
    return order
                
def create_features(participants, teamWin):
    autofillRating = getAutofillRating(champion_stats)

    url = "https://ddragon.leagueoflegends.com/cdn/13.24.1/data/en_US/champion.json"
    r = requests.get(url=url)
    r = r.json()
    keys = [int(championData["key"]) for championName, championData in r['data'].items()]
    keyToIndex = {key: index for index, key in enumerate(keys)}

    blueTotalMasteryLevel = 0
    blueTotalMasteryPoints = 0
    redTotalMasteryLevel = 0
    redTotalMasteryPoints = 0
    blueWinRatings = []
    redWinRatings = []

    blueSynergyScore, redSynergyScore = getSynergy(participants, synergyMatrix, keyToIndex)

    for i, participant in enumerate(participants):
        champId = participant["championId"]
        pos = i % 5
        if i < 5:
            blueTotalMasteryLevel += participant["masteryLevel"]
            blueTotalMasteryPoints += participant["masteryPoints"]
            blueWinRatings.append(getWinLoss(champId, pos, autofillRating, champion_stats))
        else:
            redTotalMasteryLevel += participant["masteryLevel"]
            redTotalMasteryPoints += participant["masteryPoints"]
            redWinRatings.append(getWinLoss(champId, pos, autofillRating, champion_stats))

    topDelta, jgDelta, midDelta, botDelta, supDelta = getDeltas(participants, counterMatrix, keyToIndex)

    features = {
        'blueTopRating': blueWinRatings[0],
        'blueJgRating': blueWinRatings[1],
        'blueMidRating': blueWinRatings[2],
        'blueBotRating': blueWinRatings[3],
        'blueSupRating': blueWinRatings[4],
        'redTopRating': redWinRatings[0],
        'redJgRating': redWinRatings[1],
        'redMidRating': redWinRatings[2],
        'redBotRating': redWinRatings[3],
        'redSupRating': redWinRatings[4],
        'blueAvgMasteryLevel': blueTotalMasteryLevel / 5,
        'blueAvgMasteryPoints': blueTotalMasteryPoints / 5,
        'redAvgMasteryLevel': redTotalMasteryLevel / 5,
        'redAvgMasteryPoints': redTotalMasteryPoints / 5,
        'blueSynergyScore': blueSynergyScore,
        'redSynergyScore': redSynergyScore,
        'topDelta': topDelta,
        'jgDelta': jgDelta,
        'midDelta': midDelta,
        'botDelta': botDelta,
        'supDelta': supDelta,
        'teamWin': teamWin
    }
    
    return features

def predict_model(features, display_df):
    df = getDataset(drive_service)
    new_data_point = pd.DataFrame(features, index=[0])
    df = pd.concat([df, new_data_point], ignore_index=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop("teamWin", axis=1))

    print(df["teamWin"])

    # Select only the last row for prediction
    X = X[-1:]
    y = df["teamWin"].iloc[-1]

    predictions = model.predict(X)
    display_df = display_df.style.applymap(lambda x: style_columns(x, index_val=0), subset=pd.IndexSlice[:,['summonerNameBlue','championBlue']]).applymap(lambda x: style_columns(x, index_val=1), subset=pd.IndexSlice[:,['summonerNameRed','championRed']])
    
    st.dataframe(
        display_df,
        column_config={
            "summonerNameBlue": "Summoner :Blue side",
            "championBlue" : "Champion : Blue side",
            "role": "Position",
            "summonerNameRed": "Summoner : Red side",
            "championRed" : "Champion : Red side",
        },
        hide_index =True,
    )

    binary_predictions = (predictions > 0.5).astype(int)
    predicted_outcome = "Blue side wins" if binary_predictions[0][0] == 0 else "Red side wins"
    actual_outcome = "Blue side wins" if (y == 0).any() else "Red side wins"

    return predicted_outcome, actual_outcome
    #print(f'Prediction: {binary_predictions[0][0]}')
    #print(f'Actual: {y}')


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

#The User Interface
st.header("League of Legends game outcome predictor")
player_name = st.text_input("Enter Summoner Name: ")
#region_name = st.text_input("Enter Region: ")

if st.button("Search Player"): 
    status, r_live, summoner_Id = live_game_status(player_name, headers)
    if (status):
        st.write("**Player currently in a game**")
        st.write("**Current Game Prediction**")
        summoners_live, champions_live= live_game_data(r_live)
        participants_live = []
        order = champion_positions(r_live)
        display_df = pd.DataFrame(
            {
                "summonerNameBlue": [],
                "championBlue": [],
                "role": [],
                "summonerNameRed": [],
                "championRed": []
            }
        )
        try:
            for i in range(10):
                x = order[i]
                masteryLevel, masteryPoints = getMastery(champions_live[x][1], summoners_live[x][1])
                time.sleep(0.8)
                data = {
                    'summonerId': summoners_live[x][0],
                    'puuid': summoners_live[x][1],
                    'champion': champions_live[x][0],
                    'championId': champions_live[x][1],
                    'masteryLevel': masteryLevel,
                    'masteryPoints': masteryPoints
                }
                if (summoners_live[x][0] == summoner_Id):
                    index = x
                    print(index)
                
                roles = {0: 'TOP', 1: 'JUNGLE', 2: 'MIDDLE', 3: 'BOTTOM', 4: 'UTILITY'}
                if (x < 5):
                    new_data = {'summonerNameBlue': summoners_live[x][2], 'championBlue': champions_live[x][0], 'role': roles[i], 'summonerNameRed': summoners_live[order[i+5]][2], 'championRed': champions_live[order[i+5]][0]}

                    df_new_rows = pd.DataFrame([new_data])
                    print(new_data)
                    display_df = pd.concat([display_df,df_new_rows], ignore_index=True)
                print(data)
                participants_live.append(data)
        except:
            print("Could not retrieve match data")
    

        teamWin = 0 #doesn't matter as we do not know outcome yet
        features = create_features(participants_live, teamWin)
        predcicted_outcome_live, actual_outcome_live = predict_model(features, display_df)

        st.write("Predicted outcome: "+ predcicted_outcome_live)
    
    st.subheader("Past 5 game predictions")
    for a in range (5):
        st.write(f'**Game {str(a+1)} :**')
        matchId, player_puuid = getMatchId(player_name, headers, a)
        url4 = f"https://europe.api.riotgames.com/lol/match/v5/matches/{matchId}"
        r = requests.get(url=url4, headers=headers)
        time.sleep(1)

        participants = []
        index = 0
        if(r.status_code == 200):
            r = r.json()
            champions = getChampions(r)
            summoners = getMatchParticipants(r)
            display_df = pd.DataFrame(
                {
                    "summonerNameBlue": [],
                    "championBlue": [],
                    "role": [],
                    "summonerNameRed": [],
                    "championRed": []
                }
            )
            try:
                for x in range(10):
                    masteryLevel, masteryPoints = getMastery(champions[x][1], summoners[x][1])
                    time.sleep(0.8)
                    data = {
                        'summonerId': summoners[x][0],
                        'puuid': summoners[x][1],
                        'champion': champions[x][0],
                        'championId': champions[x][1],
                        'role': champions[x][2],
                        'win': champions[x][3],
                        'masteryLevel': masteryLevel,
                        'masteryPoints': masteryPoints
                    }
                    if (summoners[x][1] == player_puuid):
                        index = x
                        print(index)
                    
                    if (x < 5):
                        new_data = {'summonerNameBlue': summoners[x][2], 'championBlue': champions[x][0], 'role': champions[x][2], 'summonerNameRed': summoners[x+5][2], 'championRed': champions[x+5][0]}

                        df_new_rows = pd.DataFrame([new_data])
                        display_df = pd.concat([display_df,df_new_rows], ignore_index=True)
                   
                    participants.append(data)
            except:
                print("Could not retrieve match data")
        else:
            print("Could not load match data") 

        teamWin = getMatchWinner(participants)
        features = create_features(participants, teamWin)
        predicted_outcome, actual_outcome = predict_model(features, display_df)
        st.write("Predicted outcome: "+ predicted_outcome)
        st.write("Actual outcome: "+ actual_outcome)
       