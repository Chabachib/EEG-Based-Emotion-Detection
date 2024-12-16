import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import *

# Title and Intro
st.title("Real-Time EEG Emotion Detection")
st.write("Welcome to the Real-Time EEG Emotion Detection Simulator!")

# Explanation Section
st.write(
    """
### What is This Simulation About?
This simulation processes real-time EEG (Electroencephalography) data to analyze the emotional state of a person. 
Using machine learning, we can predict emotions like happiness, sadness, or relaxation based on the brain's electrical activity.

In this demo, you can:
- Select an EEG data file.
- Simulate the real-time processing of EEG signals.
- Visualize the EEG signal for multiple channels.
- See the predicted emotion for each segment of data.
"""
)

# Adding media
st.write("### Meet the Person Behind the EEG Data")
video_placeholder = st.empty()

st.write("### Select an EEG Data File:")
file_options = [f"s{str(i).zfill(2)}.dat" for i in range(1, 33)]
selected_file = st.selectbox("Choose a file:", file_options)

st.write("File loaded successfully. Starting simulation...")

# Load video or image for visualization
person_photo_path = f"media/1.mp4"

video_placeholder.video(person_photo_path)

# Simulate with media
if st.button("Start Simulation"):
    file_path = f"data_preprocessed_python/{selected_file}"
    st.write(f"Reading file: {selected_file}")

    eeg_data = read_eeg_data(file_path)

    if eeg_data is not None:
        # EEG data processing
        trial_count, channel_count, data_length = eeg_data.shape
        st.write(
            f"Data Shape: {trial_count} trials, {channel_count} channels, {data_length} data points"
        )

        plot_placeholder = st.empty()
        prediction_placeholder = st.empty()

        for trial in range(trial_count):
            st.write(f"Simulating Trial {trial + 1}...")

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

                if t % (4 * 128) == 0 and t >= 512:
                    time_segment = t // (4 * 128)
                    eeg_data_segment = eeg_data[trial, :, t - 512 : t]
                    prediction = make_prediction(trial, time_segment, eeg_data_segment)
                    prediction_placeholder.markdown(f"### **{prediction}**")

                time.sleep(0.5)

            st.write(f"Trial {trial + 1} completed.")

        st.success("Simulation completed.")
