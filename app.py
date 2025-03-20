import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model import build_model, train_model
from utils import plot_sample_digits, plot_training_history

st.title("Handwritten Digit Recognition")

st.write("Upload your `train.csv` and `test.csv` files.")

# File uploaders for train and test data
train_file = st.file_uploader("Upload train.csv", type="csv")
test_file = st.file_uploader("Upload test.csv", type="csv")

if train_file is not None and test_file is not None:
    # Load the data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    st.write("### Data Information")
    st.write("Training Data Shape:", train_data.shape)
    st.write("Test Data Shape:", test_data.shape)
    st.write("First 5 labels:", train_data['label'].head().values)

    # Plot sample digits
    st.write("### Sample Digits")
    fig_sample = plot_sample_digits(train_data)
    st.pyplot(fig_sample)

    # Prepare data
    X = train_data.drop('label', axis=1).values
    y = train_data['label'].values
    X = X / 255.0
    test_data = test_data / 255.0

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Training set shape:", X_train.shape)
    st.write("Validation set shape:", X_val.shape)

    # Build the model
    model = build_model(input_shape=(784,))
    st.write("### Model Summary")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    if st.button("Train Model"):
        # Train the model
        history = train_model(model, X_train, y_train, X_val, y_val)
        st.success("Training completed!")
        st.write(f"**Validation Accuracy:** {history.history['val_accuracy'][-1] * 100:.2f}%")

        # Plot training history
        st.write("### Training History")
        fig_history = plot_training_history(history)
        st.pyplot(fig_history)
