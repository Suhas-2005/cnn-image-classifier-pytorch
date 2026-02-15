# CIFAR-10 Image Classifier

This project implements a Convolutional Neural Network (CNN) for classifying images into the CIFAR-10 dataset categories. The model can classify images from URLs into one of ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Features

- CNN model optimized for CIFAR-10 classification
- Support for loading and processing images from URLs
- Confidence scores for predictions
- Top-5 predictions for each image
- Training progress visualization
- Model checkpointing (saves best model)

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the CNN architecture
- `train.py`: Training script for the model
- `predict.py`: Script for making predictions on new images
- `requirements.txt`: List of required Python packages

## Training the Model

To train the model, run:

```bash
python train.py
```

The script will:
1. Download the CIFAR-10 dataset
2. Train the model for 30 epochs
3. Save the best model as 'cifar10_best.pth'
4. Generate a training curve plot

## Making Predictions

To use the trained model for predictions:

```python
from predict import CIFAR10Predictor

# Initialize the predictor
predictor = CIFAR10Predictor(model_path='cifar10_best.pth')

# Make prediction on an image URL
image_url = "your_image_url_here"
predictions = predictor.predict(image_url)

# Print predictions
for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['class']} - {pred['confidence']*100:.2f}% confidence")
```

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with batch normalization
- Max pooling after each convolutional layer
- Dropout for regularization
- 2 fully connected layers
- Softmax output layer

## Performance

The model typically achieves around 75-80% accuracy on the CIFAR-10 test set after training.

## Notes

- The model expects input images to be RGB
- Images are automatically resized to 32x32 pixels
- The model returns confidence scores for all predictions
- GPU acceleration is automatically used if available 