# Importing necessary libraries
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, filename="error_log.log", filemode="a")

# Load dataset and preprocess
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]

# Sort the dataset by emotional and pleasant tags and reset index for proper handling
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split data for emotion-based categories
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Enhanced Emotion-to-Song Mapping and Dynamic Song Recommendation Based on Emotion Intensity
def fun(list_emotions, emotion_intensity):
    data = pd.DataFrame()
    
    # Adjust sample size based on emotion intensity
    if emotion_intensity == 'high':
        sample_size = [50, 40, 30, 20]
    else:
        sample_size = [30, 20, 15, 10]  # Default sample sizes
    
    # Emotion-based filtering for song recommendations
    for idx, emotion in enumerate(list_emotions):
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=min(sample_size[idx], len(df_neutral)))], ignore_index=True)
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=min(sample_size[idx], len(df_angry)))], ignore_index=True)
        elif emotion == 'Fear':
            data = pd.concat([data, df_fear.sample(n=min(sample_size[idx], len(df_fear)))], ignore_index=True)
        elif emotion == 'Happy':
            data = pd.concat([data, df_happy.sample(n=min(sample_size[idx], len(df_happy)))], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=min(sample_size[idx], len(df_sad)))], ignore_index=True)
    
    return data

# Function to preprocess the list of emotions to ensure uniqueness
def pre(list_emotions):
    unique_list = list(Counter(list_emotions).keys())
    return unique_list

# Optimized Pipeline for Recommendation with Error Handling
def optimize_pipeline(list_emotions, emotion_intensity):
    results = []
    if list_emotions:  # Ensure list_emotions is not empty
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(fun, [emotion], emotion_intensity) for emotion in list_emotions]
            for future in futures:
                result = future.result()
                if not result.empty:  # Check that the result is not empty
                    results.append(result)
    else:
        logging.error("No emotions detected to generate recommendations.")
    
    if results:
        return pd.concat(results)
    else:
        logging.error("No valid data to concatenate in the results.")
        return pd.DataFrame()  # Return an empty dataframe if no results

# Real-Time Playlist Update Mechanism
def real_time_playlist_update(list_emotions, emotion_intensity):
    # Logic to dynamically update the playlist in real-time based on detected emotions
    updated_playlist = optimize_pipeline(list_emotions, emotion_intensity)
    return updated_playlist

# CNN Model architecture for emotion detection
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load pre-trained weights without displaying the error on the page
try:
    model.load_weights('model.h5')
except ValueError as e:
    logging.error(f"Error loading weights: {str(e)}")

# Emotion categories
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Streamlit UI Setup
st.markdown("<h2 style='text-align: center; color: black'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
list_emotions = []

# Initialize recommended_songs with an empty DataFrame
recommended_songs = pd.DataFrame()

# Real-time emotion detection from webcam
with col2:
    if st.button('SCAN EMOTION (Click here)'):
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])  # Create a Streamlit image object to update frames
        while len(list_emotions) < 20:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            # For each detected face, predict the emotion and append to the list
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[max_index], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                list_emotions.append(emotion_dict[max_index])
            
            # Update the live camera view in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color to RGB for Streamlit
            frame_window.image(frame)
        
        # Process the list of emotions for unique values
        list_emotions = pre(list_emotions)
        cap.release()
        cv2.destroyAllWindows()

# Select emotion intensity
emotion_intensity = st.selectbox('Select Emotion Intensity', ['normal', 'high'])

# Generate real-time song recommendations based on the detected emotions and selected intensity
if list_emotions:
    recommended_songs = real_time_playlist_update(list_emotions, emotion_intensity)
    if recommended_songs.empty:
        st.write("No songs available for the detected emotions.")
else:
    st.write("No emotions detected. Please try scanning again.")

# Display the recommended songs with clickable links
if not recommended_songs.empty:
    st.write("Recommended Songs:")
    for _, row in recommended_songs.iterrows():
        st.markdown(f"<h4 style='text-align: center;'><a href='{row['link']}'>{row['name']}</a></h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center;'><i>{row['artist']}</i></h5>", unsafe_allow_html=True)
