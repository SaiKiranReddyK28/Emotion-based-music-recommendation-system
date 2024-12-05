import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import speech_recognition as sr
import pyttsx3
from keras.models import load_model

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Load the pretrained emotion detection model
model = load_model("model.h5")

# Emotion mapping
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Function to process voice commands
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        tts_engine.say("Listening for a command...")
        tts_engine.runAndWait()
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            tts_engine.say("Sorry, I did not understand. Please try again.")
            tts_engine.runAndWait()
            return None

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

# Header
st.markdown("<h2 style='text-align: center;'>Voice-Enabled Emotion-Based Music Recommendation</h2>", unsafe_allow_html=True)

# Initialize variables
list_emotions = []
emotion_count = {emotion: 0 for emotion in emotion_dict.values()}

# Listen for user commands
if st.button("Enable Voice Commands"):
    command = listen_for_command()
    if command:
        if "scan emotion" in command:
            st.write("Voice Command: Scan Emotion")
            tts_engine.say("Starting emotion scanning.")
            tts_engine.runAndWait()

            cap = cv2.VideoCapture(0)
            frame_window = st.image([])  # Create a Streamlit image object to update frames
            while len(list_emotions) < 20:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y + h, x:x + w]
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

            tts_engine.say("Emotion scanning completed.")
            tts_engine.runAndWait()

        elif "recommend songs" in command:
            st.write("Voice Command: Recommend Songs")
            tts_engine.say("Fetching recommended songs.")
            tts_engine.runAndWait()

            # Simulate song recommendation logic
            recommended_songs = pd.DataFrame({
                "name": ["Song A", "Song B"],
                "artist": ["Artist X", "Artist Y"],
                "link": ["https://example.com/songA", "https://example.com/songB"],
            })

            st.write("### Recommended Songs")
            for _, row in recommended_songs.iterrows():
                st.markdown(f"<h4><a href='{row['link']}' target='_blank'>{row['name']} by {row['artist']}</a></h4>", unsafe_allow_html=True)
            
            tts_engine.say("Here are your song recommendations.")
            tts_engine.runAndWait()

        else:
            tts_engine.say("Unknown command. Please say 'Scan Emotion' or 'Recommend Songs'.")
            tts_engine.runAndWait()

# Display emotion intensity meter graph
if list_emotions:
    st.image(plot_intensity_meter(emotion_count), use_column_width=True)
