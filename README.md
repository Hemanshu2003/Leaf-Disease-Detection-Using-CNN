# Leaf Disease Detection Using CNN

This project focuses on detecting diseases in plant leaves using Convolutional Neural Networks (CNNs). By leveraging deep learning techniques, the model identifies different types of leaf diseases from images, which can aid in efficient and timely intervention.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Model Architecture](#model-architecture)
5. [How to Run](#how-to-run)
6. [Results](#results)
8. [License](#license)

---

## Introduction

Plant diseases significantly impact agricultural yield and productivity. This project automates the detection of leaf diseases by analyzing images with CNN-based models. It aims to help farmers and agriculturists by providing a reliable and scalable solution for disease detection.

Predicting disease for 4 species of plants :
- Corn 
- Grapes
- Potato 
- Tomato

---

## Dataset

The dataset consists of images of healthy and diseased leaves, categorized into various classes such as:

- Healthy
- Powdery Mildew
- Rust
- Other specific diseases

### Preprocessing Steps

1. **Resizing**: All images are resized to ensure uniformity.
2. **Augmentation**: Techniques like rotation, flipping, and zooming were applied to increase dataset variability.
3. **Normalization**: Pixel values were scaled to the range `[0, 1]` for faster convergence during training.

---

## Technologies Used

- **Programming Language**: Python
- **Framework**: TensorFlow / Keras
- **Notebook Environment**: Jupyter Notebook
- **Visualization**: Matplotlib, Seaborn

---

## Model Architecture

The Convolutional Neural Network consists of the following layers:

1. **Convolutional Layers**: Extract spatial features from images.
2. **Pooling Layers**: Reduce dimensionality while retaining essential information.
3. **Fully Connected Layers**: Combine features for classification.
4. **Output Layer**: Predicts the class of the input image.

---

## How to Run

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Libraries: TensorFlow, Keras, NumPy, Matplotlib, Pandas, OpenCV

### Steps to Execute

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/leaf-disease-detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd leaf-disease-detection
   ```

3. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook Main.ipynb
   ```

5. Follow the instructions in the notebook to train and test the model.

---

## Results

### Evaluation Metrics

- **Accuracy**: Achieved % accuracy on the test set.
- **Precision, Recall, and F1-Score**: Detailed evaluation metrics for each class.

### Visualizations

1. **Training and Validation Loss**:

   ![Training and Validation LossPlot](https://github.com/Hemanshu2003/Leaf-Disease-Detection-Using-CNN/blob/main/Output1.png)

2. **Confusion Matrix**:

   ![Training and Validation Accuracy Plot](https://github.com/Hemanshu2003/Leaf-Disease-Detection-Using-CNN/blob/main/Output2.png)

---

## Future Enhancements

- Expand the dataset to include more leaf diseases.
- Optimize the CNN architecture for faster inference.
- Implement a mobile-friendly interface for real-time predictions.

---


Feel free to contribute by opening issues or submitting pull requests! Together, we can make agricultural technology smarter and more accessible.
