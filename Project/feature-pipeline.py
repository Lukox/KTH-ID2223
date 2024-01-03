import os
import modal
import json
from io import BytesIO
import pickle
import pandas as pd
import requests
import time
from io import StringIO
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload


LOCAL=False


if LOCAL == False:
   stub = modal.Stub("league")
   image = modal.Image.debian_slim().pip_install("pandas","google-auth", "google-auth-oauthlib", "google-auth-httplib2", "google-api-python-client")

   @stub.function(image=image, timeout = 2000, secrets=[modal.Secret.from_name("my-googlecloud-secret-2"), modal.Secret.from_name("RIOT_API_KEY"),])
   def f():
       global headers
       headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en,pl-PL;q=0.9,pl;q=0.8,en-US;q=0.7",
        "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://developer.riotgames.com",
        "X-Riot-Token": os.environ["RIOT_API_KEY"]
    }
       g()

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

def getParticipants(drive_service):
    files = getFile("dataset.pkl", drive_service)
    if files:
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        content = BytesIO(request.execute())
        df = pickle.load(content)
        return df
    else:
        print(f"File dataset.pkl not found.")

def getPuuids(summonerIds):
    puuids = []
    for i in range(min(5, len(summonerIds))):
        url_summoners = f"https://euw1.api.riotgames.com/lol/summoner/v4/summoners/{summonerIds[i]}"
        r = requests.get(url=url_summoners, headers=headers)
        if r.status_code != 200:
            print(r.status_code)
            if r.status_code == 429:
                print("LIMIT")
                continue
        puuids.append(r.json()["puuid"])
        time.sleep(0.8)
    return puuids

def getMatchIds(rank, numOfGames):
    url = f"https://euw1.api.riotgames.com/lol/league-exp/v4/entries/RANKED_SOLO_5x5/{rank}/I?page=1"
    r = requests.get(url=url, headers=headers)
    players = r.json()
    summonerIds = [player["summonerId"] for player in players]
    puuids = getPuuids(summonerIds)
    totalGames = 0
    match_ids = []
    for puuid in puuids:
        if totalGames > numOfGames:
            break
        url4 = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start=0&count=1"
        r = requests.get(url=url4, headers=headers)
        for id in r.json():
            match_ids.append(id)
        print(id)
        totalGames+=20
        time.sleep(0.8)
    return match_ids


def getNewMatches():
    ranks = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
    total_games = 10

    rank_percentages = {
        "IRON": 7.5,
        "BRONZE": 20,
        "SILVER": 19,
        "GOLD": 19,
        "PLATINUM": 19,
        "EMERALD": 13,
        "DIAMOND": 3,
        "MASTER": 0.5,
        "GRANDMASTER": 0.04,
        "CHALLENGER": 0.02
    }
    rank_games = {rank: int(total_games * percentage / 100) for rank, percentage in rank_percentages.items()}
    allMatchIds = []
    for rank in ranks:
        allMatchIds.append(getMatchIds(rank, rank_games[rank]))
    matchIds = [item for sublist in allMatchIds for item in sublist]
    return matchIds

def updateWinMatrix(champion_stats, data):
    # Process the data and update the statistics
    for entry in data:
        champion_name = entry[0]
        champion_id = entry[1]
        position = entry[2]
        win = entry[3]
        if position == '':
          print("No Position")
          continue

        # Initialize the dictionary if the champion is not present
        if champion_id not in champion_stats:
            champion_stats[champion_id] = {'TOP': {'wins': 0, 'losses': 0},
                                           'JUNGLE': {'wins': 0, 'losses': 0},
                                           'MIDDLE': {'wins': 0, 'losses': 0},
                                           'BOTTOM': {'wins': 0, 'losses': 0},
                                           'UTILITY': {'wins': 0, 'losses': 0}}

        # Update the statistics based on the win status
        if win:
            champion_stats[champion_id][position]['wins'] += 1
        else:
            champion_stats[champion_id][position]['losses'] += 1
    return champion_stats

def updateSynergyMatrix(matrix, data, matrixSize, keyToIndex):
    # Process the data and update the matrix for the first 5 champions
    for i in range(5):
        for j in range(5):
            if i != j:  # Exclude the champion itself
                try:
                  win = data[i][3]
                  teammateKey = data[j][1]
                  championKey = data[i][1]
                  matrix[keyToIndex[championKey]][keyToIndex[teammateKey]]['wins' if win else 'losses'] += 1
                except:
                  continue

    # Process the data and update the matrix for the remaining 5 champions
    for i in range(5, 10):
        for j in range(5, 10):
            if i != j:  # Exclude the champion itself
                try:
                  win = data[i][3]
                  teammateKey = data[j][1]
                  championKey = data[i][1]
                  matrix[keyToIndex[championKey]][keyToIndex[teammateKey]]['wins' if win else 'losses'] += 1
                except:
                  continue
    return matrix

def updateCounterMatrix(matrix, data, matrixSize, keyToIndex):
    for i in range(5):
      for j in range(5,10):
        try:
          if data[i][2] == data[j][2]:
            win = data[i][3]
            enemyKey = data[j][1]
            championKey = data[i][1]
            matrix[keyToIndex[championKey]][keyToIndex[enemyKey]]['wins' if win else 'losses'] += 1
            matrix[keyToIndex[enemyKey]][keyToIndex[championKey]]['losses' if win else 'wins'] += 1
        except:
          continue
    return matrix

def getMastery(championId, puuid):
  url = f"https://euw1.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{championId}"
  r = requests.get(url=url, headers=headers)
  if(r.status_code == 200):
    r = r.json()
    return  r["championLevel"], r["championPoints"]
  else:
    print("Error getting mastery")
    return 0, 0
  
def getChampions(r):
    champions = [[participant["championName"],participant["championId"],participant["teamPosition"], participant["win"]] for participant in r["info"]["participants"]]
    return champions

def getMatchParticipants(r):
  participants = [[participant["summonerId"], participant["puuid"]] for participant in r["info"]["participants"]]
  return participants

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

def getMatchWinner(matchData):
  if(matchData[0]["win"] == True):
    return 0
  else:
    return 1
  
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

def saveData(df, drive_service):
    files = getFile("NewData.csv", drive_service)
    if files:
        content = df.to_csv(index=False)
        file_metadata = {
            "name": 'NewData.csv',
            "mimeType": "text/csv"
        }
        file_id = files[0]['id']
        media = MediaIoBaseUpload(BytesIO(content.encode()), mimetype='text/csv')
        drive_service.files().update(body=file_metadata,fileId=file_id, media_body = media).execute()


def saveToDrive(data, filename, drive_service):
    files = getFile(filename, drive_service)
    if files:
        content = BytesIO()
        pickle.dump(data, content)
        file_metadata = {
            "name": filename,
        }
        content.seek(0)
        file_id = files[0]['id']
        media = MediaIoBaseUpload(content, mimetype='application/octet-stream')
        drive_service.files().update(body=file_metadata,fileId=file_id, media_body = media).execute()

def g():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    drive_service = build('drive', 'v3', credentials=credentials)

    newMatches = getNewMatches()

    url = "https://ddragon.leagueoflegends.com/cdn/13.24.1/data/en_US/champion.json"
    r = requests.get(url=url)
    r = r.json()
    keys = [int(championData["key"]) for championName, championData in r['data'].items()]

    keyToIndex = {key: index for index, key in enumerate(keys)}

    participants = []
    df = getDataset(drive_service)
    champion_stats = getWinMatrix(drive_service)
    synergyMatrix = getSynergyMatrix(drive_service)
    counterMatrix = getCounterMatrix(drive_service)

    count = 0
    for matchId in newMatches:
        url4 = f"https://europe.api.riotgames.com/lol/match/v5/matches/{matchId}"
        r = requests.get(url=url4, headers=headers)
        time.sleep(0.8)

        if(r.status_code == 200):
            r = r.json()
            champions = getChampions(r)
            summoners = getMatchParticipants(r)

            champion_stats = updateWinMatrix(champion_stats, champions)
            synergyMatrix = updateSynergyMatrix(synergyMatrix, champions, len(keys), keyToIndex)
            counterMatrix = updateCounterMatrix(counterMatrix, champions, len(keys), keyToIndex)

            participant = []
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
                    participant.append(data)
            except:
                continue
        participants.append(participant)
        count+=1
    autofillRating = getAutofillRating(champion_stats)
    saveToDrive(counterMatrix, "NewCounterMatrix.pkl", drive_service)
    saveToDrive(synergyMatrix, "NewSynergyMatrix.pkl", drive_service)
    saveToDrive(champion_stats, "NewWinRateMatrix.pkl", drive_service)
    
    for matchData in participants:
        blueTotalMasteryLevel = 0
        blueTotalMasteryPoints = 0
        redTotalMasteryLevel = 0
        redTotalMasteryPoints = 0
        blueWinRatings = []
        redWinRatings = []

        teamWin = getMatchWinner(matchData)
        blueSynergyScore, redSynergyScore = getSynergy(matchData, synergyMatrix, keyToIndex) 

        for i, participant in enumerate(matchData):
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

        topDelta, jgDelta, midDelta, botDelta, supDelta = getDeltas(matchData, counterMatrix, keyToIndex)

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
        new_data_point = pd.DataFrame(features, index=[0])
        df = pd.concat([df, new_data_point], ignore_index=True)
    print (df)
    saveData(df, drive_service)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()