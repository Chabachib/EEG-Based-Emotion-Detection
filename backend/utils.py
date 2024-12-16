import numpy as np
import pyeeg as pe
import pickle as pickle
import pandas as pd
import math
import keras
import streamlit as st


from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from keras.api.models import load_model


channels = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]
band = [4, 8, 12, 16, 25, 45]
window_size = 256
step_size = 16
sample_rate = 128
subjectList = ["01", "02", "03", "04", "05", "06"]


def load_cnn_model():
    return load_model("model/cnn_92_acc.h5")


def normalize_standarize(data):
    return normalize(data)


def std_scale(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)


def FFT_Processing(data):
    meta = []

    start = 0
    while start + window_size <= data.shape[1]:
        meta_data = []

        for channel in channels:
            segment = data[channel, start : start + window_size]
            fft_features, _ = pe.bin_power(segment, band, sample_rate)
            meta_data.extend(fft_features)

        meta.append(meta_data)
        start += step_size

    meta = np.array(meta)
    return meta


def map_emotion(arousal, valence):
    if arousal > 0.5 and valence > 0.5:
        return "Happy"
    elif arousal > 0.5 and valence <= 0.5:
        return "Angry"
    elif arousal <= 0.5 and valence > 0.5:
        return "Relaxed"
    else:
        return "Sad"


def make_prediction(trial, time_segment, data_segment):
    model = load_cnn_model()

    fft_data = FFT_Processing(data_segment)

    X_norm = normalize(fft_data)

    X_scaled = std_scale(X_norm)

    X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    predictions = model.predict(X)

    arousal_prediction = "High" if predictions[0][0] > 0.5 else "Low"
    valence_prediction = "High" if predictions[0][1] > 0.5 else "Low"

    return f"Segment {time_segment + 1}: Arousal: {arousal_prediction}, Valence: {valence_prediction} | Detected Emotion : {map_emotion(predictions[0][0], predictions[0][1])}"


def read_eeg_data(file_path):
    try:
        with open(file_path, "rb") as f:
            loaded_file = pickle.load(f, encoding="latin1")
            data = np.array(loaded_file["data"])
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
