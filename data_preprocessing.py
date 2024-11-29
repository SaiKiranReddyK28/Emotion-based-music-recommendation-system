# Import additional libraries for plotting
import matplotlib.pyplot as plt
import io
import base64

# Function to plot real-time emotion intensity meter
def plot_intensity_meter(emotion_count):
    # Emotion count dictionary passed will track intensity based on the frequency of emotions detected
    emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    intensities = [emotion_count.get(emotion, 0) for emotion in emotions]
    
    fig, ax = plt.subplots()
    ax.bar(emotions, intensities, color=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown'])
    ax.set_title('Emotion Intensity Meter')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Intensity')

    # Convert plot to PNG and encode it to base64 for display in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"

# Function to update emotion intensity dynamically
def update_emotion_intensity(emotion_dict, detected_emotions):
    # Calculate the frequency of each emotion detected
    emotion_count = dict.fromkeys(emotion_dict.values(), 0)
    for emotion in detected_emotions:
        if emotion in emotion_count:
            emotion_count[emotion] += 1
    return emotion_count

# Optimizing the transition of detected emotions to playlist recommendations
def optimized_transition(prev_recommendations, new_recommendations, emotion_intensity):
    # Apply intensity-based weighted blending of previous and new recommendations
    intensity_factor = 0.7 if emotion_intensity == 'high' else 0.5
    combined_recommendations = pd.concat([prev_recommendations, new_recommendations]).drop_duplicates().reset_index(drop=True)
    
    # Adjust weighting: higher intensity gives more weight to the new recommendations
    combined_recommendations['intensity'] = combined_recommendations['pleasant'] * intensity_factor
    combined_recommendations = combined_recommendations.sort_values(by='intensity', ascending=False)
    return combined_recommendations

# Real-Time Playlist Update Mechanism with Optimized Transition
def real_time_playlist_update_optimized(list_emotions, emotion_intensity, previous_recommendations):
    updated_playlist = optimize_pipeline(list_emotions, emotion_intensity)
    if not updated_playlist.empty and not previous_recommendations.empty:
        # Blend the new playlist with the previous one based on the emotion intensity
        updated_playlist = optimized_transition(previous_recommendations, updated_playlist, emotion_intensity)
    return updated_playlist

# Streamlit UI Setup for displaying emotion intensity meter
st.markdown("<h2 style='text-align: center; color: black'><b>Emotion Based Music Recommendation with Intensity Meter</b></h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
list_emotions = []
emotion_count = {emotion: 0 for emotion in emotion_dict.values()}  # Initialize emotion count
previous_recommendations = pd.DataFrame()  # Track previously generated recommendations

# Real-time emotion detection from webcam and updating intensity meter
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
                emotion_detected = emotion_dict[max_index]
                list_emotions.append(emotion_detected)
                emotion_count = update_emotion_intensity(emotion_dict, list_emotions)  # Update intensity meter
            
            # Update the live camera view in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color to RGB for Streamlit
            frame_window.image(frame)
        
        # Process the list of emotions for unique values
        list_emotions = pre(list_emotions)
        cap.release()
        cv2.destroyAllWindows()

# Select emotion intensity
emotion_intensity = st.selectbox('Select Emotion Intensity', ['normal', 'high'])

# Display emotion intensity meter graph
st.image(plot_intensity_meter(emotion_count), use_column_width=True)

# Generate real-time song recommendations based on the detected emotions and selected intensity
if list_emotions:
    recommended_songs = real_time_playlist_update_optimized(list_emotions, emotion_intensity, previous_recommendations)
    previous_recommendations = recommended_songs  # Update the previous recommendations with the new ones
    
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
