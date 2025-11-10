import streamlit as st
import sounddevice as sd
import numpy as np
import time
import matplotlib.pyplot as plt

# Settings
DURATION = 1   # seconds per sample
FS = 44100     # sampling rate
SILENT_THRESHOLD = 30
WARNING_THRESHOLD = 55

# Streamlit Page Config
st.set_page_config(page_title="Noise Detector", page_icon="🔊", layout="centered")
st.title("📢 Noise Control Detector for Library/Classroom")

# Status Box
status_placeholder = st.empty()
chart_placeholder = st.empty()

# Function to get noise level
def get_noise_level(indata):
    rms = np.sqrt(np.mean(indata**2))
    db = 20 * np.log10(rms + 1e-10)
    return db

# Store last N values for chart
noise_history = []

def callback(indata, frames, time_info, status):
    global noise_history
    db_level = get_noise_level(indata)
    
    # Update status message
    status_msg = "✅ Silent"
    color = "green"
    if db_level > WARNING_THRESHOLD:
        status_msg = "❌ Too Noisy!"
        color = "red"
    elif db_level > SILENT_THRESHOLD:
        status_msg = "⚠️ Warning!"
        color = "orange"
    
    status_placeholder.markdown(f"<h2 style='color:{color};text-align:center;'>{status_msg} ({db_level:.2f} dB)</h2>", unsafe_allow_html=True)

    # Update chart
    noise_history.append(db_level)
    if len(noise_history) > 50:  # Keep last 50 samples
        noise_history.pop(0)

    fig, ax = plt.subplots()
    ax.plot(noise_history, label="Noise Level (dB)")
    ax.axhline(y=SILENT_THRESHOLD, color='orange', linestyle='--', label="Warning Threshold")
    ax.axhline(y=WARNING_THRESHOLD, color='red', linestyle='--', label="Too Noisy Threshold")
    ax.set_ylabel("Decibels (dB)")
    ax.set_xlabel("Time (samples)")
    ax.legend()
    chart_placeholder.pyplot(fig)

# Run Streamlit Noise Detector
st.write("🎤 Listening... Speak or clap to test.")

try:
    with sd.InputStream(callback=callback, channels=1, samplerate=FS):
        while True:
            time.sleep(0.5)
except KeyboardInterrupt:
    st.write("Stopped.")
