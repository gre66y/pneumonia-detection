# Pneumonia Detection using Deep Learning

This project implements pneumonia detection from chest X-ray images using VGG19 and ResNet50 models.

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset from Kaggle and extract it. 
Link for the dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
The dataset should extracted in the following structure if not structure it as so:
   ```
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

## Features

- Automatic skipping of corrupted/truncated images
- Model evaluation with precision, recall, F1 score, and accuracy
- Confusion matrix visualization
- Training history plots
- Model comparison between VGG19 and ResNet50

## Model Architecture

Both models use transfer learning with pre-trained weights from ImageNet:
- Base models: VGG19 and ResNet50
- Additional layers: Global Average Pooling and Dense layers
- Binary classification output with sigmoid activation

## Output

The script will generate:
- Evaluation metrics for both models
- Confusion matrices
- Training history plots
- Saved model files (.h5 format)