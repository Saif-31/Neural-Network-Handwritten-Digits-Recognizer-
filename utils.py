# -*- coding: utf-8 -*-
"""utils

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vdxcOUqjJXWcGeeZju-L1r_BDtTZZnKJ
"""

import matplotlib.pyplot as plt

def plot_sample_digits(train_data):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        # Reshape the first 10 images (assuming each image is 28x28 pixels)
        img = train_data.drop('label', axis=1).iloc[i].values.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {train_data['label'].iloc[i]}")
        ax.axis('off')
    plt.tight_layout()
    return fig

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.tight_layout()
    return fig