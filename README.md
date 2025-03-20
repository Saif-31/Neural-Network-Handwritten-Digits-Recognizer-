# Handwritten Digit Recognition Streamlit App

This repository contains a Streamlit application for training a neural network to recognize handwritten digits using TensorFlow.

## Features
- **Data Upload:** Upload your `train.csv` and `test.csv` files.
- **Data Visualization:** Display sample digit images and basic data information.
- **Model Training:** Build and train a neural network with a simple user interface.
- **Training History:** Visualize training accuracy and loss over epochs.

## Repository Structure

```
handwritten-digit-streamlit/
├── app.py             # Main Streamlit application file
├── model.py           # Functions to build and train the TensorFlow model
├── utils.py           # Utility functions (e.g. plotting)
├── requirements.txt   # Required Python packages
└── README.md          # Repository documentation
