# Real-Time Nepali Number Plate Recognition

## Overview

This project focuses on developing a **Real-Time Nepali Number Plate Recognition** system. It leverages advanced techniques in object detection, segmentation, and optical character recognition (OCR) to identify and recognize Nepali number plates written in the Devanagari script. The system is built using YOLOv7 for object localization and HOG+SVM for character segmentation and classification.

## Objective

To create an real-time system for recognizing Nepali vehicle number plates using state-of-the-art object detection (YOLOv7) and character recognition (SVM with HOG features).

## Problem Statement

Nepal lacks an automated system for accurately identifying vehicle number plates in real-time, making tasks such as **traffic management**, **law enforcement**, and **toll collection** inefficient and prone to human error. The unique structure of Nepali number plates, written in the Devanagari script, adds complexity to the recognition process.

## literature Review

### Automatic License Plate Recognition Using Deep Learning
-  **Concept**: Utilized CNN-based models for number plate detection and recognition.
   Metrics: Achieved 98.3% accuracy on Latin-based number plates.

YOLO-Based Vehicle Number Plate Detection and Recognition
  Concept: Implemented YOLOv3 for real-time object detection with a focus on number plate localization.
  Metrics: Precision of 97.5% with real-time processing capability (~30 FPS).

Nepali Number Plate Recognition Using SVM
  Concept: Applied SVMs to recognize segmented characters from Nepali license plates(learning-based methods for character recognition)
  Metrics: Accuracy of 85% on Nepali number plates.


## System Architecture
   ![System Architecture](./Real-time-Licence-Plate-Recognition/datasets_link/image.png)



### YOLOv7 for Object Localization
- **Description**: YOLOv7, a state-of-the-art object detection model, is used to detect and localize the number plate within the image.
- **Justification**: YOLOv7 was selected for its high speed and accuracy, making it suitable for real-time processing, especially in tasks involving small objects like number plates.

### HOG + Contour + SVM for Character Recognition
- **HOG (Histogram of Oriented Gradients)**: Used for feature extraction from the segmented characters.
- **Contour Detection**: For isolating characters from the number plate.
- **SVM (Support Vector Machine)**: Classifies segmented characters into their respective Devanagari or numeric categories.

## Datasets

### Vehicle Number Plate Dataset (Nepal)
- **Source**: Kaggle
- **Description**: Contains over 8000 images of Nepali number plates captured from various angles and lighting conditions.
- **Annotations**: Each image includes bounding box annotations for the number plate.

### Nepali Number Plate Characters Dataset
- **Source**: Kaggle
- **Description**: Contains over 800 images of individual Nepali characters extracted from number plates.
- **Classes**: Covers 35 Devanagari characters (vowels, consonants, numbers).

## Training and Fine-Tuning

- **Data Preparation**: The dataset is split into training (70%), validation (20%), and testing (10%) sets. Data augmentation techniques (random flips, rotations, color adjustments) are applied to enhance robustness.
- **YOLOv7 Fine-Tuning**: The pre-trained YOLOv7 model is fine-tuned using custom Nepali dataset annotations to localize the number plates.
- **HOG + SVM Training**: After character segmentation using contours, HOG features are extracted and an SVM classifier is trained to recognize individual characters.

### Hyperparameters
- **Learning Rate**: 0.001
- **Batch Size**: 16 or 32
- **Epochs**: 20-40
- **Optimizer**: Adam for faster convergence.

## Challenges Faced
- **Limited Dataset Availability**: Lack of comprehensive datasets for all characters, especially custom fonts.
- **Character Segmentation**: Some characters were not segmented correctly, leading to misclassification.
- **Dealing with Skew and Noise**: Handling skewed plates and noisy images posed challenges.

## Future Enhancements

1. **Integration of Deep Learning for OCR**: Implementing CRNN (Convolutional Recurrent Neural Network) for more robust character recognition, especially for complex fonts and lighting variations.
2. **Multi-Language Support**: Expanding the system to support number plates from different countries and regions.

## Real-World Applications

- **Traffic Management**: Automating vehicle tracking and managing traffic congestion.
- **Law Enforcement**: Assisting in identifying stolen vehicles and tracking suspects.
- **Toll Collection**: Seamless, automatic toll collection without the need for physical tickets.
- **Security and Surveillance**: Enhancing access control in gated communities, airports, and government buildings.

## Problems and System Robustness

- The system struggles with **non-standard fonts**, **unclear characters**, and **challenging lighting conditions**. Further refinements are necessary to enhance accuracy and performance in real-world scenarios.

## Conclusion

This project successfully implements a real-time system for Nepali number plate recognition using YOLOv7 for object detection and HOG+SVM for character recognition. However, to make the system more robust and deployable in real-world conditions, further fine-tuning and dataset expansion are required. you can check outputs on output_folders.

  
