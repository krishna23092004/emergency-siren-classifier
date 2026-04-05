import streamlit as st
import numpy as np
import librosa
import os
import tempfile
import tensorflow as tf

# --- Page Config ---
st.set_page_config(page_title="Siren Detector", page_icon="🚨")

st.title("🚨 Emergency Siren Detection")
st.write("Upload an audio clip to identify Ambulance, Firetruck, or Traffic sounds.")

# --- Constants ---
MODEL_PATH = "model.keras"   # Make sure this file exists in repo
CLASS_NAMES = ['ambulance', 'firetruck', 'traffic']


# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


# --- Feature Extraction ---
def extract_audio_features(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

    try:
        audio, sample_rate = librosa.load(temp_path, res_type='kaiser_fast')

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled.reshape(1, -1)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- UI ---
model = load_model()

uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if uploaded_audio is not None and model is not None:
    st.audio(uploaded_audio)

    with st.spinner("Analyzing sound..."):
        try:
            features = extract_audio_features(uploaded_audio)

            prediction_probs = model.predict(features)[0]

            predicted_idx = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_idx] * 100

            st.divider()

            result_label = CLASS_NAMES[predicted_idx].upper()

            if result_label in ['AMBULANCE', 'FIRETRUCK']:
                st.error(f"### 🚨 {result_label} DETECTED")
            else:
                st.success(f"### 🚦 {result_label} SOUND")

            st.metric("Confidence", f"{confidence:.2f}%")

            with st.expander("See full breakdown"):
                for name, prob in zip(CLASS_NAMES, prediction_probs):
                    st.write(f"**{name.capitalize()}:** {prob*100:.2f}%")
                    st.progress(float(prob))

        except Exception as e:
            st.error(f"Error processing audio: {e}")