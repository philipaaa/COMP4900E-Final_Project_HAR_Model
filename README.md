COMP4900E-Final_Project_HAR_Model
Deterministic Human Activity Recognition on QNX

    This project implements a human activity recognition (HAR) system using smartphone accelerometer data and a lightweight machine learning model.

Overview
    - Sensor data collected via smartphone (SensorLog)
    - Data processed into fixed-size windows
    - Statistical features extracted per window
    - Logistic regression model used for classification
    - Designed for real-time deployment on QNX


How to Run
    python src/train_model.py
    or :
    python3 src/train_model.py

Notes
    - Raw data is not included due to privacy considerations
    - The trained model is exported for deployment on a QNX system
    - Real-time performance is evaluated through timing of feature extraction and inference
