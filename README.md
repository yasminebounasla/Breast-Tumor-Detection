**Breast Tumor Classification Project**:

I built a Convolutional Neural Network (CNN) to classify breast tumors as benign or malignant using medical images.

**Project Overview**:
Dataset: I used a dataset of breast tumor images. The images are divided into two categories: benign (0) and malignant (1).

Preprocessing: I resized all images to 224x224 pixels and normalized their pixel values to improve the model's performance.

Model: The CNN I created has layers that extract features from the images, reduce the dimensions, and finally classify them as benign or malignant.

Training: I split the data into training and testing sets (80% training, 20% testing) and applied oversampling to handle class imbalance. I trained the model in batches of 128 images for 2 epochs.

Testing: After training, I evaluated the model on the test dataset to check its accuracy.

Prediction: I added a function that lets me predict if a tumor is benign or malignant for a specific image.

Tools and Libraries:

TensorFlow
NumPy
Pillow (PIL)
scikit-learn
imbalanced-learn (for handling class imbalance)


**Outcome**:

The model can predict if a tumor is benign or malignant based on images. It helps with early diagnosis using deep learning.

