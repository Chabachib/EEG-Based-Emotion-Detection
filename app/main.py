import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import *

st.set_page_config(page_title="Real-Time EEG Emotion Detection", page_icon="🧠")
st.title("Real-Time EEG Emotion Detection")
st.write("Welcome to the Real-Time EEG Emotion Detection Simulator!")
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

st.write("### Meet the Person Behind the EEG Data")
video_placeholder = st.empty()

st.write("### Select an EEG Data File:")
file_options = [f"s{str(i).zfill(2)}.dat" for i in range(2, 7)]
selected_file = st.selectbox("Choose a file:", file_options)

# Load video or image for visualization based on the selected file
person_photo_path = f"media/{selected_file.replace('.dat', '.mp4')}"
video_placeholder.video(person_photo_path)

st.write("File loaded successfully. Starting simulation...")

# Initialize statistics table
if "statistics" not in st.session_state:
    st.session_state.statistics = []

# Simulate with media
if st.button("Start Simulation"):
    file_path = f"../data_preprocessed_python/{selected_file}"
    st.write(f"Reading file: {selected_file}")

    eeg_data = read_eeg_data(file_path)

    if eeg_data is not None:
        # EEG data processing
        trial_count, channel_count, data_length = eeg_data.shape
        st.write(
            f"Data Shape: {trial_count} trials, {channel_count} channels, {data_length} data points"
        )

        plot_placeholder = st.empty()  # For accumulated signal
        segment_plot_placeholder = st.empty()  # For 4-second sliding window
        prediction_placeholder = st.empty()
        stats_placeholder = st.empty()

        for trial in range(trial_count):
            st.write(f"Simulating Trial {trial + 1}...")

            for t in range(0, data_length, 64):
                # Plot accumulated signals
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

                # Plot non-accumulated 4-second sliding window (updated every 0.5 seconds)
                if t >= 512:  # Ensure we have at least 4 seconds of data
                    fig_segment, ax_segment = plt.subplots(figsize=(15, 6))
                    start_idx = max(0, t - 512)  # 4 seconds = 512 data points (128 Hz * 4)
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

                    # Make prediction and update statistics every 4 seconds
                    if t % (4 * 128) == 0:
                        time_segment = t // (4 * 128)
                        eeg_data_segment = eeg_data[trial, :, start_idx:end_idx]
                        prediction = make_prediction(trial, time_segment, eeg_data_segment)
                        prediction_placeholder.markdown(f"### **{prediction}**")

                        # Update statistics table
                        st.session_state.statistics.append({
                            "Trial": trial + 1,
                            "Segment": time_segment + 1,
                            "Start Time (s)": start_idx / 128,
                            "End Time (s)": end_idx / 128,
                            "Prediction": prediction
                        })

                        # Display statistics table
                        stats_placeholder.table(st.session_state.statistics)

                time.sleep(0.5)  # Update every 0.5 seconds

            st.write(f"Trial {trial + 1} completed.")

        st.success("Simulation completed.")