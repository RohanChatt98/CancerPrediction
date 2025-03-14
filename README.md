# Cancer Prediction Model

## Overview

The **Cancer Prediction** project aims to predict the likelihood of cancer (malignant or benign) in patients based on a variety of diagnostic features. Using machine learning algorithms, specifically a **Random Forest Classifier** model, the project trains a model on a dataset of features to classify whether a tumor is malignant or benign. The model is designed to be deployed in a continuous integration/continuous deployment (CI/CD) pipeline, ensuring a seamless workflow from training to testing and deployment.

---

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description

This project uses a dataset with various features extracted from medical images of tumors to predict if a tumor is **benign** or **malignant**. The dataset consists of several numeric attributes that describe the characteristics of the tumor, such as its size, shape, and texture. The project pipeline involves several stages:

1. **Data Preprocessing**: Clean and prepare the dataset for training.
2. **Model Training**: Train a machine learning model on the preprocessed data.
3. **Model Evaluation**: Evaluate the model's performance using accuracy, precision, recall, and F1 score.
4. **Model Saving**: Save the trained model for future inference and use.
5. **Model Testing**: Verify the model's performance through unit tests.

---

## Features

- **Data Preprocessing**: Cleans and prepares the data using normalization and encoding techniques.
- **Model Training**: Trains the model using a **Random Forest Classifier**.
- **Model Evaluation**: Evaluates model performance using standard metrics.
- **Model Testing**: Verifies model prediction accuracy and saving/loading functionality.
- **CI/CD Pipeline**: Continuous integration and deployment using AWS CodeBuild and GitHub for automating the training, testing, and deployment process.

---

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CancerPrediction.git
   cd CancerPrediction
