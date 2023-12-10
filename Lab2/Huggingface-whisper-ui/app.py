import io
import streamlit as st
from transformers import pipeline
from pytube import YouTube
from pydub import AudioSegment
import os
import math
import pandas as pd
import pysrt 
import requests
import json
from moviepy.editor import VideoFileClip

#downloads our model from Whisper
pipe = pipeline(model=("Lukox/whisper-pl"))

#creates global 'len' variable which represents the length of the youtube video in seconds
if 'len' not in st.session_state:
    st.session_state['len'] = 0

#function that uses the model to transcribe the audio and output it as text
def transcribe(audio):
    text = pipe(audio)["text"]
    return text

#uses ffmpeg to split the youtube video into equal 5 second mp4 files (except for the last video, depending on len). 
#the output video will have the split_%03d.mp4 format. This is because our model transcribes videos up to 30 seconds
#so dividing it into equal short videos bypasses this limit and allows for easier subtitle making
def split_vid(name):
    os.system("ffmpeg -i "+name+ " -segment_time 00:00:05 -f segment -y -reset_timestamps 1 -c copy split_%03d.mp4")
    st.write("Completed downloading video")

def transcription_vids(name):
    os.system("ffmpeg -i "+name+ " -segment_time 25 -f segment -y -reset_timestamps 1 -c copy transcription_%03d.mp4")

#creates and appends the texts to the polish srt file
def add_subtitles_srt(text):
    srt = open("file.srt", "a")
    srt.write(text)
    srt.close()

#creates and appends the texts to the translated srt file
def add_subtitles_srt_trans(text):
    srt = open("file_trans.srt", "a")
    srt.write(text)
    srt.close()

#creates and appends the texts to the polish vtt file. This file is not used, but could be used 
#instead of the srt files as it has a similar format and ffmpeg can also use it to add subtitles
def add_subtitles_vtt(text):
    vtt = open("file.vtt", "a")
    vtt.write(text)
    vtt.close()

#converts seconds into the appropriate time format for srt and vtt files. The punc is to differentiate
#the files, as they have a slight difference in format. Srt uses ',' to separate milliseconds while
#vtt uses '.' to separate milliseconds
def seconds_to_string(seconds, punc):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_string = "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))+ punc +"000"

    return time_string  

#converts the audio of the split vids to the appriopriate format
def convert(audio_content):
    #st.audio(audio_content, format="audio/wav", start_time=0)
    audio = AudioSegment.from_file(audio_content, format="mp4")
    audio.export(audio_content, format="mp3")
    return audio_content

#uses ffmpeg to combine the youtube video (mp4) with the srt file, and outputs a polish substitled mp4 video
def create_video_player_pl():
    os.system("ffmpeg -i file.mp4 -vf subtitles=file.srt:force_style='Fontsize=12' output_video.mp4")
    return "output_video.mp4"

#uses ffmpeg to combine the youtube video (mp4) with the srt file, and outputs a translated substitled mp4 video
def create_video_player_trans():
    os.system("ffmpeg -i file.mp4 -vf subtitles=file_trans.srt:force_style='Fontsize=12' output_video_trans.mp4")
    return "output_video_trans.mp4"

#format (add zeroes) depending on time component
def format_time_component(value, is_milliseconds=False):
    padded_value = str(value).zfill(2)
    return padded_value if not is_milliseconds else str(value).zfill(3)

#format the hours, minutes, seconds and milliseconds to the correct format
def format_time(sub_time):
    hours = format_time_component(sub_time.hours)
    minutes = format_time_component(sub_time.minutes)
    seconds = format_time_component(sub_time.seconds)
    millis = format_time_component(sub_time.milliseconds, is_milliseconds=True)
    return hours, minutes, seconds, millis

#uses pysrt to create a table/dataframe which contains the transcribed texts and their timestamps. The timestamps are all 5 second chunks
def timestamps(vid_name):
    df = pd.DataFrame(columns = ['start','end','text'])
    subs = pysrt.open(vid_name)

    objects = []
    for sub in subs:
        
        start_hours, start_minutes, start_seconds, start_millis = format_time(sub.start)
        end_hours, end_minutes, end_seconds, end_millis = format_time(sub.end)
        objects.append([sub.text, f'{start_hours}:{start_minutes}:{start_seconds}.{start_millis}', f'{end_hours}:{end_minutes}:{end_seconds}.{end_millis}'])

    for object in objects:
        srt_to_df = {
        'start': [object[1]],
        'end': [object[2]], 
        'text': [object[0]] 
        }

        df = pd.concat([df, pd.DataFrame(srt_to_df)])

    return df

#uses the DeepL translation API to translate the transcribed texts. The API key comes from creating a 
#DeepL account and allows a monthly usage of 500,000 characters. The translation will be in json
def translate(text, language):
    
    api_key = 'cd3a8e66-7491-2da1-50cc-6bcb525be92d:fx'
    api_url = 'https://api-free.deepl.com/v2/translate'

    data = {
        'text': text,
        'source_lang': 'PL',
        'target_lang': language,
        'auth_key': api_key
    }

    response = requests.post(api_url, data=data)
    result = json.loads(response.text)
    translated_text = result['translations'][0]['text']

    return translated_text

#uses the pytube libary to download videos, renames it to file.mp4 for simplicity and calls the split_vid method to divide the video into 5 second files
def download_youtube_audio(youtube_url):
    try:
        yt = YouTube(youtube_url)
        downloaded_video = (yt.streams
            .filter(progressive=True, file_extension='mp4')
            .get_highest_resolution().download()
        )
        os.rename(downloaded_video, "file.mp4")
        
        split_vid("file.mp4")
        transcription_vids("file.mp4")
        return yt.length
    finally:
        print("Video downloaded")

def transcribe_text(len, option):
    text = ""
    trans_text = ""
    language = option[:2]
    for i in range(math.ceil(len/25)):
        if (i < 10):
            file_name = "transcription_00"+str(i)+".mp4"
        elif (i < 100):
            file_name = "transcription_0"+str(i)+".mp4"
        else:
            file_name = "transcription_"+str(i)+".mp4"

        audio = convert(file_name)
        transcription = transcribe(audio)
        transcription_trans = translate(transcription, language)
        text = text + transcription
        trans_text = trans_text + transcription_trans

    return text, trans_text    

#transcribes each split file (5 seconds videos) and combines them to create the whole transcription of the video. The srt and vtt files are also created 
def combine(len, option):
    time = 0
    text = ""
    trans_text = ""
    language = option[:2]
    add_subtitles_vtt("WEBVTT\n\n")
    #for loop calls for each split video file. In the case there are less files than loops, the try should catch that error and continue without crashing
    for i in range(len):
        try:
            if (i < 10):
                file_name = "split_00"+str(i)+".mp4"
            elif (i < 100):
                file_name = "split_0"+str(i)+".mp4"
            else:
                file_name = "split_"+str(i)+".mp4"
            
            seconds = VideoFileClip(file_name).duration
            audio = convert(file_name)
            transcription = transcribe(audio)
            transcription_trans = translate(transcription, language)
            text = text + transcription
            trans_text = trans_text + transcription_trans
            #adding the transcriptions to the srt and vtt files in the correct formats:
            #srt: 
            # 1
            # 00:00:00,000 ---> 00:00:05,000
            # "transcription text" 

            #vtt:
            # WEBVTT (starts with this and doesn't contain numbers to indicate current dialogue)
            # 00:00:00.000 ---> 00:00:05.000
            # "transcription text" 
            subtitles_srt = str(i+1)+"\n" + seconds_to_string(time,",")+ " --> " + seconds_to_string(time+seconds,",") +"\n" +transcription +"\n\n"
            subtitles_vtt = seconds_to_string(time,".")+ " --> " + seconds_to_string(time+seconds,".") +"\n" +transcription +"\n\n"
            subtitles_srt_trans = str(i+1)+"\n" + seconds_to_string(time,",")+ " --> " + seconds_to_string(time+seconds,",") +"\n" +transcription_trans +"\n\n"
            add_subtitles_srt(subtitles_srt)
            add_subtitles_vtt(subtitles_vtt)
            add_subtitles_srt_trans(subtitles_srt_trans)
            time = time +seconds
            os.remove(file_name)
            print(i)
        except:
            print("finished transcription")
            break

    

st.header("Add subtitles to Polish youtube videos")
#allows user to input a youtube url
st.write("Please enter an appropriate youtube url and click the 'Confirm url' button to confirm. Please wait until the download is completed before proceeding to the transcription")
st.write("Please remember that some youtube videos are private and cannot be downloaded. The program will display an error id that is the case.")
youtube_url = st.text_input("Enter YouTube URL: ")

#button confirms the url and downloads the video
if st.button("Confirm url"):
    st.session_state.len = download_youtube_audio(youtube_url)

option = st.selectbox(
   "What language do you want the subtitles to be in?",
   ("EN - English",
    "BG - Bulgarian",
    "CS - Czech",
    "DA - Danish",
    "DE - German",
    "EL - Greek", 
    "ES - Spanish",
    "ET - Estonian",
    "FI - Finnish",
    "FR - French",
    "HU - Hungarian",
    "ID - Indonesian",
    "IT - Italian",
    "JA - Japanese",
    "KO - Korean",
    "LT - Lithuanian",
    "LV - Latvian",
    "NB - Norwegian",
    "NL - Dutch",
    "PT - Portuguese", 
    "RO - Romanian",
    "RU - Russian",
    "SK - Slovak",
    "SL - Slovenian",
    "SV - Swedish",
    "TR - Turkish",
    "UK - Ukrainian",
    "ZH - Chinese"),
   placeholder="Select language...",
)
st.write("Please wait until the language is loaded before transcribing")
st.write("If subtitles are not translated, this may be caused by reaching the usage limit of the DeepL translation API. Please try again with a smaller video or after the 9th of the next month")
#button that transcribes the video and creates all the videos, texts and tables
if st.button("Transcribe"):
    st.subheader("Original video:")     
    st.video("file.mp4")
    combine(st.session_state.len, option)   
    print(len)

    f = open("file.srt", "r")
    print(f.read()) 
    f = open("file_trans.srt", "r")
    print(f.read())    
    language_name = option[4:]
    mass_text , mass_trans_text = transcribe_text(st.session_state.len, option)

    #Display transcription results in text
    st.header("Transcription Results")
    st.subheader("Polish Transcription")
    st.write(mass_text)
    df = timestamps("file.srt")
    st.dataframe(df)

    st.subheader(language_name+" transcription")
    st.write(mass_trans_text)
    df_eng = timestamps("file_trans.srt")
    st.dataframe(df_eng)
    

    #create the subtitled videos and display them
    output_vid_pl = create_video_player_pl()
    output_video_trans = create_video_player_trans()
    st.subheader("Polish subtitles")
    st.video(output_vid_pl)
    st.subheader(language_name+" subtitles")
    st.video(output_video_trans)

    #remove all the files 
    os.remove("output_video.mp4")
    os.remove("output_video_trans.mp4")
    os.remove("file.mp4")
    os.remove("file.vtt")
    os.remove("file.srt")
    os.remove("file_trans.srt")