# Cat-Dog Classifier
This project involves building a machine learning model using TensorFlow and Keras to classify images of cats and dogs. The model is trained on a dataset of labeled images and can predict whether a new image contains a cat or a dog.

# Introduction
The Cat-Dog Classifier project aims to develop a convolutional neural network (CNN) that can distinguish between images of cats and dogs. The model is built using TensorFlow and Keras, popular libraries for machine learning and deep learning.

# Dataset
The dataset used for training and testing the model is the Kaggle Dogs vs. Cats dataset. It contains 10,000 labeled images of cats and dogs, with an equal number of images for each class.

# Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:
#### Convolutional layers with ReLU activation and max pooling
#### Fully connected (dense) layers
#### Dropout layers for regularization

# Training
The model is trained using the Adam optimizer and categorical cross-entropy loss. The training process includes:

#### Data augmentation to increase the diversity of the training set
#### Early stopping to prevent overfitting
#### Model checkpointing to save the best model

# Evaluation
The model is evaluated on a separate test set using accuracy, precision, recall, and F1-score as metrics. Confusion matrices and ROC curves are also generated to visualize the performance.

# Results
After training, the model achieves an accuracy of approximately 95% on the test set. The performance may vary depending on the training conditions and dataset splits.
