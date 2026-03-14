EEG-ECG Feature Fusion and Emotion Classification

Project Overview
This repository provides a simple EEG-ECG feature fusion pipeline for classification tasks. The code extracts handcrafted features from EEG and ECG signals, fuses the two modalities, splits the fused features into training and validation sets, and trains a machine learning classifier for evaluation.

Files
1. FeatureFusion.py
   - Loads EEG data from an NPZ file and ECG data from a CSV file.
   - Extracts EEG features, including band power and entropy-related features.
   - Extracts ECG features, including RMSSD, SDNN, LF power, HF power, and LF/HF ratio.
   - Standardizes and fuses EEG and ECG features.
   - Splits the fused dataset into training and validation sets.

2. Train.py
   - Trains a machine learning classifier on the fused features.
   - Evaluates the model using Accuracy, Precision, Recall, and F1-score.
   - The current version uses a Random Forest classifier.

Environment
Python 3.9 or above is recommended.

Dependencies
Install the required packages with:

pip install -r requirements.txt

Required Input Data
Before running the code, prepare the following files:
1. EEG data in NPZ format, containing:
   - X: EEG samples
   - y: labels
2. ECG data in CSV format, containing:
   - ECG_Signal: ECG signal column

Important Notes
1. The current code contains hard-coded local paths, for example:
   - C:/Users/cyl/Desktop/naodian/001.npz
   - C:/Users/cyl/Desktop/xindian/data/001.csv
   You should modify these paths before running the code.

2. Train.py depends on variables generated in FeatureFusion.py:
   - X_train
   - X_validate
   - Y_train
   - Y_validate
   Therefore, Train.py is not fully standalone in its current form.

3. For open-source release, it is recommended to:
   - Move data paths into configurable variables or command-line arguments.
   - Save the processed training and validation sets to files before training.
   - Refactor Train.py so it can independently load processed data.

How to Run
Option 1: Run in an interactive environment
1. Execute FeatureFusion.py first.
2. Execute Train.py in the same Python session.

Option 2: Refactor before release
A better open-source structure is:
1. FeatureFusion.py generates and saves processed features.
2. Train.py reads the saved features and performs training independently.

Example Workflow
Step 1: Feature extraction and fusion
python FeatureFusion.py

Step 2: Model training and evaluation
python Train.py


Suggested Repository Structure
project/
├── FeatureFusion.py
├── Train.py
├── requirements.txt
├── README.txt
└── data/
    ├── 001.npz
    └── 001.csv

License
Please choose an open-source license before publishing the repository, such as MIT, Apache-2.0, or GPL-3.0.

Contact
You may add author, affiliation, and contact information here if needed.
