# EEG-Based-Emotion-Detection

This repository contains the code and resources for an EEG-based emotion detection system. The project leverages deep learning models to classify emotions using EEG signals from the DEAP dataset. The goal is to detect emotions such as valence and arousal, which are categorized into "high" and "low" states.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## Introduction

Emotion detection through EEG signals is a significant area of research in affective computing and human-computer interaction. This project implements machine learning techniques to classify emotions based on EEG data from the DEAP dataset.

## Dataset

The [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) is a widely-used dataset for emotion analysis, containing EEG and peripheral physiological signals of 32 participants as they watched music videos. Each participant's data includes ratings for arousal, valence, and dominance.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Chabachib/EEG-Based-Emotion-Detection.git
   cd EEG-Based-Emotion-Detection
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation:**
   - Ensure the DEAP dataset is downloaded and placed in the appropriate directory.
   - Preprocess the data using the provided scripts in the `notebooks` directory.

2. **Model Training:**
   - Use the Jupyter notebooks in the `notebooks` directory to train the emotion detection models.

3. **Evaluation:**
   - Evaluate the models using the test datasets and visualize the results stored in the `figures` directory.

## Project Structure

The repository is organized as follows:

- **app/**: Contains the main application code.
- **csv-files/**: Includes CSV files generated during data preprocessing.
- **figures/**: Stores visualizations and figures related to data analysis and model performance.
- **models/**: Contains trained machine learning models.
- **notebooks/**: Includes Jupyter notebooks for data exploration, preprocessing, model training, and evaluation.
- **EEG_Based_Emotion_Detection_Report.pdf**: Detailed project report.
- **Presentation.pptx**: Project presentation slides.
- **requirements.txt**: Lists the Python dependencies required to run the project.

## Results

The project report (`EEG_Based_Emotion_Detection_Report.pdf`) and presentation (`Presentation.pptx`) provide detailed insights into the methodology, experiments, and findings. Key results include:

- Data preprocessing techniques applied to the DEAP dataset.
- Machine learning models trained for emotion classification.
- Evaluation metrics and performance analysis.
