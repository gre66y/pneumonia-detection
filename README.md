# Pneumonia Detection using VGG19 and ResNet50

## Overview
Pneumonia is a severe respiratory disease that affects millions worldwide. Early and accurate detection is crucial for timely treatment. This project leverages deep learning techniques, particularly **VGG19** and **ResNet50**, to classify chest X-ray images as **Normal** or **Pneumonia**.

## Problem Statement
Traditional pneumonia detection through X-ray examination is time-consuming and dependent on expert radiologists. This project aims to automate the process using **Convolutional Neural Networks (CNNs)** to improve efficiency, accuracy, and accessibility.

## Objectives
- Preprocess and organize the chest X-ray dataset into **training, validation, and test** sets.
- Implement **VGG19** and **ResNet50** models for pneumonia classification.
- Evaluate model performance using **accuracy, precision, recall, and F1-score**.
- Visualize learning curves to analyze model training and validation trends.
- Compare the two models to determine the best-performing architecture for pneumonia detection.

## Dataset
The dataset consists of labeled **chest X-ray images**, categorized into two classes:
- **Normal**: Healthy lung X-rays.
- **Pneumonia**: X-rays with pneumonia indications.

The dataset is structured into three folders:
- **Train**: Used for model training.
- **Validation**: Used for hyperparameter tuning.
- **Test**: Used for final model evaluation.

## Methodology
### 1. Data Preprocessing
- Convert grayscale images to **RGB format** to align with pre-trained model requirements.
- Resize images to **224x224 pixels**.
- Apply **normalization** (scaling pixel values between 0 and 1).
- Augment data with transformations like **random flipping, rotation, and zooming** to improve generalization.

### 2. Model Development
Both **VGG19** and **ResNet50** architectures are used for classification:
- **VGG19**: A deep CNN with a uniform structure, effective for extracting complex patterns.
- **ResNet50**: A residual network designed to mitigate the vanishing gradient problem, improving training efficiency.

#### Model Architecture Modifications:
- Removed top layers from pre-trained models.
- Added a **Global Average Pooling (GAP)** layer.
- Included **fully connected dense layers** with ReLU activation.
- Final **sigmoid activation layer** for binary classification.

### 3. Training and Evaluation
- Models trained using the **Adam optimizer**.
- Implemented **early stopping and learning rate scheduling** to optimize training.
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Confusion Matrix Analysis**

## Results and Insights
### Performance Metrics
| Model   | Accuracy | Precision | Recall | F1 Score |
|---------|---------|----------|--------|----------|
| VGG19   | 85.58%  | 87.00%   | 93.59% | 89.02%   |
| ResNet50| 78.37%  | 84.74%   | 79.74% | 82.17%   |

- **VGG19 performed better** in accuracy, precision, and recall, making it a more suitable model for pneumonia detection.
- **ResNet50 struggled** with training but still showed promising results.
- **Confusion matrix analysis** highlighted a few misclassifications, indicating room for further improvement.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Pneumonia-Detection.git
   cd Pneumonia-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the models:
   ```bash
   python main.py
   ```
4. Evaluate model performance:
   ```bash
   python evaluation.py
   ```

## Future Enhancements
- **Fine-tuning** the models by unfreezing additional layers.
- **Using ensemble learning** to combine predictions from multiple architectures.
- **Expanding the dataset** to improve generalizability.
- **Deploying as a web or mobile application** for real-world usage.
- **Integrating Explainable AI (XAI)** techniques like Grad-CAM for interpretability.

## Conclusion
This project demonstrates the effectiveness of deep learning models for automated pneumonia detection using chest X-rays. The **VGG19 model outperformed ResNet50**, making it the preferred choice for deployment in a real-world setting. Future work can enhance model accuracy, scalability, and integration into medical diagnostics.
