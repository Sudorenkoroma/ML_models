# ML Models Portfolio

This repository contains various machine learning models implemented. Each model demonstrates different approaches for solving classification tasks using deep learning techniques.

## Files in the Repository

1. **Conv_NN**  
   This file implements a Convolutional Neural Network (CNN) to classify the Fashion MNIST dataset. The model consists of multiple
   convolutional and pooling layers, followed by dense layers. The CNN architecture is designed to extract features from images and perform classification.  
   - **Key Features**:
     - Uses `Conv2D` and `MaxPooling2D` layers.
     - Dropout for regularization.
     - Early stopping to prevent overfitting.
   - **Dataset**: Fashion MNIST

3. **vgg16_base_model**  
   This file demonstrates transfer learning using the pre-trained VGG16 model on the Fashion MNIST dataset. The pre-trained model's top
   layers are replaced with custom dense layers to adapt to the new classification task. Fine-tuning is applied to the deeper layers of VGG16 to improve performance.
   - **Key Features**:
     - Uses the pre-trained `VGG16` model without top layers.
     - Fine-tuning of the last few layers.
     - Data augmentation is applied to improve generalization.
   - **Dataset**: Fashion MNIST

5. **RNN_LSTM_models**  
   This file contains multiple Recurrent Neural Network (RNN) models to classify movie reviews from the IMDb dataset. It includes implementations of simple RNN, LSTM, Bidirectional LSTM, and deep LSTM architectures.
   - **Key Features**:
     - Different RNN architectures like `SimpleRNN`, `LSTM`, and `Bidirectional LSTM`.
     - Text data preprocessing with tokenization and padding.
     - Early stopping and model checkpointing for optimization.
   - **Dataset**: IMDb movie reviews


1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_models.git
   cd ML_models
