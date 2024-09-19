# ML Models Portfolio

This repository contains various machine learning models implemented for job application purposes. Each model demonstrates different approaches for solving classification tasks using deep learning techniques.

## Files in the Repository

1. **Conv_NN**  
   This file implements a Convolutional Neural Network (CNN) to classify the Fashion MNIST dataset. The model consists of multiple convolutional and pooling layers, followed by dense layers. The CNN architecture is designed to extract features from images and perform classification.  
   - **Key Features**:
     - Uses `Conv2D` and `MaxPooling2D` layers.
     - Dropout for regularization.
     - Early stopping to prevent overfitting.
   - **Dataset**: Fashion MNIST

2. **vgg16_base_model**  
   This file demonstrates transfer learning using the pre-trained VGG16 model on the Fashion MNIST dataset. The pre-trained model's top layers are replaced with custom dense layers to adapt to the new classification task. Fine-tuning is applied to the deeper layers of VGG16 to improve performance.
   - **Key Features**:
     - Uses the pre-trained `VGG16` model without top layers.
     - Fine-tuning of the last few layers.
     - Data augmentation is applied to improve generalization.
   - **Dataset**: Fashion MNIST

3. **RNN_LSTM_models**  
   This file contains multiple Recurrent Neural Network (RNN) models to classify movie reviews from the IMDb dataset. It includes implementations of simple RNN, LSTM, Bidirectional LSTM, and deep LSTM architectures.
   - **Key Features**:
     - Different RNN architectures like `SimpleRNN`, `LSTM`, and `Bidirectional LSTM`.
     - Text data preprocessing with tokenization and padding.
     - Early stopping and model checkpointing for optimization.
   - **Dataset**: IMDb movie reviews

4. **GPT_text_clas**  
   This file implements a text classification model using GPT-like architecture. It processes and classifies text inputs by tokenizing and padding sequences to handle text data efficiently. The model can be fine-tuned on custom datasets and evaluated on various text classification tasks.
   - **Key Features**:
     - Uses transformer-based GPT architecture.
     - Preprocessing includes tokenization and sequence padding.
     - Capable of fine-tuning and text classification for different tasks.
   - **Dataset**: Custom text input for sentiment analysis.

5. **BERT_model_tensor**  
   This file contains a model based on BERT (Bidirectional Encoder Representations from Transformers) for text classification. BERT is a state-of-the-art model designed to handle a wide range of Natural Language Processing (NLP) tasks, including text classification, question answering, and more. In this implementation, BERT is fine-tuned for a specific text classification task.
   - **Key Features**:
     - Utilizes pre-trained BERT from TensorFlow Hub.
     - Fine-tuning on specific text classification datasets.
     - Tokenization using BERT's WordPiece tokenizer.
   - **Dataset**: Custom dataset for text classification.
