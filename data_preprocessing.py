# Importing necessary libraries
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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

# Function to select songs based on emotions
def fun(list_emotions):
    data = pd.DataFrame()
    sample_size = [30, 20, 15, 10]  # Sample sizes based on number of detected emotions
    
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

# CNN Model architecture for emotion detection
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
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

# Load pre-trained weights
model.load_weights('model.h5')

# Emotion categories
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Streamlit UI Setup
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
list_emotions = []

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

# Generate song recommendations based on the detected emotions
recommended_songs = fun(list_emotions)

# Display the recommended songs with clickable links
st.write("Recommended Songs:")
for _, row in recommended_songs.iterrows():
    st.markdown(f"<h4 style='text-align: center;'><a href='{row['link']}'>{row['name']}</a></h4>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'><i>{row['artist']}</i></h5>", unsafe_allow_html=True)
