import streamlit as st
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import *

st.set_page_config(page_title="Real-Time EEG Emotion Detection", page_icon="🧠")
st.title("Real-Time EEG Emotion Detection")
st.write("#### Welcome to the Real-Time EEG Emotion Detection Simulator!")
st.write(
    """
### What is this Simulation About?
This simulation processes real-time EEG (ElectroEncephaloGraphy) data from the DEAP Dataset to analyze the emotional state of the participant. 
Using machine and deep learning, we can predict emotions like happiness, sadness, or relaxation based on the brain's electrical activity.

In this demostration, you can:
- Select an EEG data file.
- Simulate the real-time processing of EEG signals.
- Visualize the EEG signal for multiple channels.
- See the predicted emotion for each segment of data.
"""
)

st.sidebar.title("Credits")
st.sidebar.write("- Nouh Taha CHEBCHOUB")
st.sidebar.write("- Ayoub BENALI")
st.sidebar.write("- Badr MOUAZEN")
st.sidebar.write("- El Hassan ABDELWAHED")
st.sidebar.write("- Giovanni de Marco")

st.write("### Meet the participant behind the EEG data")
video_placeholder = st.empty()

st.write("### Select an EEG Data File:")
file_options = [f"s{str(i).zfill(2)}.dat" for i in range(1, 33)]
selected_file = st.selectbox("Choose a file:", file_options)

person_photo_path = f"media/{selected_file.replace('.dat', '.mp4')}"
video_placeholder.video(person_photo_path)

def list_music_files(folder_path):
    try:
        return [file for file in os.listdir(folder_path) if file.endswith('.mp3')]
    except FileNotFoundError:
        return []

# List available music files
music_folder = "music"
music_files = list_music_files(music_folder)

st.write("### Listen to EEG-Associated Audio:")
if music_files:
    selected_audio = st.selectbox("Choose a music file:", music_files)
    audio_file_path = f"{music_folder}/{selected_audio}"
    st.audio(audio_file_path)
else:
    st.write("No audio files found in the 'music' folder.")

st.write("File loaded successfully, you can start the simulation !")

if "statistics" not in st.session_state:
    st.session_state.statistics = []

if st.button("Start Simulation"):
    file_path = f"../data_preprocessed_python/{selected_file}"
    eeg_data = read_eeg_data(file_path)

    if eeg_data is not None:
        trial_count, channel_count, data_length = eeg_data.shape
        plot_placeholder = st.empty()
        segment_plot_placeholder = st.empty()
        prediction_placeholder = st.empty()
        stats_placeholder = st.empty()

        for trial in range(trial_count):
            for t in range(0, data_length, 64):
                fig, ax = plt.subplots(figsize=(15, 6))

                for channel in channels:
                    signal = eeg_data[trial, channel, : t + 64]
                    time_axis = np.arange(len(signal)) / 128
                    ax.plot(time_axis, signal, label=f"Channel {channel + 1}")

                ax.set_title(
                    f"Trial {trial + 1} - EEG Channels (0 to {t/128:.2f} seconds)"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.legend(loc="upper right", fontsize="small")

                plot_placeholder.pyplot(fig, clear_figure=True)

                if t >= 512:
                    fig_segment, ax_segment = plt.subplots(figsize=(15, 6))
                    start_idx = max(0, t - 512)
                    end_idx = t

                    for channel in channels:
                        signal_segment = eeg_data[trial, channel, start_idx:end_idx]
                        time_axis_segment = np.arange(len(signal_segment)) / 128
                        ax_segment.plot(time_axis_segment, signal_segment, label=f"Channel {channel + 1}")

                    ax_segment.set_title(
                        f"Trial {trial + 1} - EEG Channels ({start_idx/128:.2f} to {end_idx/128:.2f} seconds)"
                    )
                    ax_segment.set_xlabel("Time (s)")
                    ax_segment.set_ylabel("Amplitude")
                    ax_segment.legend(loc="upper right", fontsize="small")

                    segment_plot_placeholder.pyplot(fig_segment, clear_figure=True)

                    if t % (4 * 128) == 0:
                        time_segment = t // (4 * 128)
                        eeg_data_segment = eeg_data[trial, :, start_idx:end_idx]
                        prediction = make_prediction(trial, time_segment, eeg_data_segment)
                        prediction_placeholder.markdown(f"### **{prediction}**")

                        st.session_state.statistics.append({
                            "Trial": trial + 1,
                            "Segment": time_segment + 1,
                            "Start Time (s)": start_idx / 128,
                            "End Time (s)": end_idx / 128,
                            "Prediction": prediction
                        })

                        stats_placeholder.table(st.session_state.statistics)

                time.sleep(0.5)

            st.write(f"Trial {trial + 1} completed.")

        st.success("Simulation completed.")