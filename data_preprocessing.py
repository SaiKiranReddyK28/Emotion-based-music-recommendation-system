# Import required libraries
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests

# Backend API URLs
METADATA_API = "http://127.0.0.1:5000/songs/metadata"
THEME_API = "http://127.0.0.1:5000/user/theme"

# Initialize session state variables
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

# Function to toggle theme
def toggle_theme():
    new_theme = "dark" if st.session_state["theme"] == "light" else "light"
    response = requests.post(THEME_API, json={"theme": new_theme})
    st.session_state["theme"] = response.json()["theme"]

# Apply theme styles dynamically
theme = st.session_state["theme"]
theme_colors = {
    "background": "#121212" if theme == "dark" else "#ffffff",
    "text": "#ffffff" if theme == "dark" else "#000000",
}

st.markdown(
    f"""
    <style>
    body {{
        background-color: {theme_colors["background"]};
        color: {theme_colors["text"]};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<h2 style='text-align: center; color: black'><b>Emotion-Based Music Recommendation with Intensity Meter</b></h2>",
    unsafe_allow_html=True
)

# Toggle button for theme
if st.button("Toggle Theme"):
    toggle_theme()

# Function to plot real-time emotion intensity meter
def plot_intensity_meter(emotion_count):
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    intensities = [emotion_count.get(emotion, 0) for emotion in emotions]

    fig, ax = plt.subplots()
    ax.bar(emotions, intensities, color=["red", "green", "blue", "yellow", "purple", "orange", "brown"])
    ax.set_title("Emotion Intensity Meter")
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Intensity")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"

# Function to update emotion intensity dynamically
def update_emotion_intensity(emotion_dict, detected_emotions):
    emotion_count = dict.fromkeys(emotion_dict.values(), 0)
    for emotion in detected_emotions:
        if emotion in emotion_count:
            emotion_count[emotion] += 1
    return emotion_count

# Optimizing the transition of detected emotions to playlist recommendations
def optimized_transition(prev_recommendations, new_recommendations, emotion_intensity):
    intensity_factor = 0.7 if emotion_intensity == "high" else 0.5
    combined_recommendations = pd.concat([prev_recommendations, new_recommendations]).drop_duplicates().reset_index(drop=True)
    combined_recommendations["intensity"] = combined_recommendations["pleasant"] * intensity_factor
    combined_recommendations = combined_recommendations.sort_values(by="intensity", ascending=False)
    return combined_recommendations

# Real-Time Playlist Update Mechanism with Optimized Transition
def real_time_playlist_update_optimized(list_emotions, emotion_intensity, previous_recommendations):
    updated_playlist = optimize_pipeline(list_emotions, emotion_intensity)
    if not updated_playlist.empty and not previous_recommendations.empty:
        updated_playlist = optimized_transition(previous_recommendations, updated_playlist, emotion_intensity)
    return updated_playlist

# Emotion Detection Setup
col1, col2, col3 = st.columns(3)
list_emotions = []
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_count = {emotion: 0 for emotion in emotion_dict.values()}
previous_recommendations = pd.DataFrame()

# Real-time emotion detection
with col2:
    if st.button("SCAN EMOTION (Click here)"):
        cap = cv2.VideoCapture(0)
        frame_window = st.image([])
        while len(list_emotions) < 20:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                emotion_detected = emotion_dict[max_index]
                list_emotions.append(emotion_detected)
                emotion_count = update_emotion_intensity(emotion_dict, list_emotions)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)
        cap.release()
        cv2.destroyAllWindows()

# Select emotion intensity
emotion_intensity = st.selectbox("Select Emotion Intensity", ["normal", "high"])

# Display emotion intensity meter graph
st.image(plot_intensity_meter(emotion_count), use_column_width=True)

# Generate real-time song recommendations
if list_emotions:
    recommended_songs = real_time_playlist_update_optimized(list_emotions, emotion_intensity, previous_recommendations)
    previous_recommendations = recommended_songs

    if recommended_songs.empty:
        st.write("No songs available for the detected emotions.")
else:
    st.write("No emotions detected. Please try scanning again.")

# Fetch and display song metadata
song_ids = [1, 2]  # Example song IDs
response = requests.get(METADATA_API, params={"song_ids": song_ids})
songs = response.json()

if songs:
    st.write("### Recommended Songs")
    for song_id, details in songs.items():
        st.markdown(
            f"""
            <div style="border: 1px solid {theme_colors['text']}; padding: 10px; margin: 10px; border-radius: 5px;">
                <img src="{details['album_art_url']}" style="width: 50px; height: 50px; float: left; margin-right: 10px;">
                <b>{details['name']}</b> by <i>{details['artist']}</i>
                <br>
                <a href="{details['link']}" target="_blank">Play Song</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.write("No songs available.")
