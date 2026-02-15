# CNN Image Classifier using PyTorch (CIFAR-10)

## Overview

This project implements an end-to-end Convolutional Neural Network (CNN) using PyTorch for multi-class image classification on the CIFAR-10 dataset.

The system includes:
- Custom CNN architecture
- Complete training pipeline
- Model checkpointing
- Inference pipeline for new images
- Training performance visualization

The trained model classifies images into one of the following 10 categories:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## Dataset

This project uses the CIFAR-10 dataset, which consists of:

- 60,000 color images
- 10 object classes
- Image size: 32x32 pixels
- 50,000 training images
- 10,000 test images

Official dataset link:
https://www.cs.toronto.edu/~kriz/cifar.html

---

## Project Structure

```
├── model.py                # CNN architecture definition
├── train.py                # Training script
├── predict.py              # Inference script
├── predict_notebook.ipynb  # Jupyter demo
├── requirements.txt        # Required dependencies
├── cifar10_best.pth        # Trained model weights
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Suhas-2005/cnn-image-classifier-pytorch.git
cd cnn-image-classifier-pytorch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the model:

```bash
python train.py
```

The training script will:
- Automatically download the CIFAR-10 dataset
- Train the CNN for 30 epochs
- Save the best-performing model as `cifar10_best.pth`
- Generate training curve plots

---

## Making Predictions

Example usage:

```python
from predict import CIFAR10Predictor

predictor = CIFAR10Predictor(model_path='cifar10_best.pth')

image_url = "your_image_url_here"
predictions = predictor.predict(image_url)

for i, pred in enumerate(predictions, 1):
    print(f"{i}. {pred['class']} - {pred['confidence']*100:.2f}% confidence")
```

The model returns:
- Top-5 predictions
- Confidence scores for each class

---

## Model Architecture

The CNN consists of:

- 3 Convolutional layers with Batch Normalization
- ReLU activation functions
- Max Pooling layers
- Dropout for regularization
- 2 Fully Connected layers
- Softmax output layer

---

## Performance

- Test Accuracy: ~75–80%
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 30

GPU acceleration is automatically used if available.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Pillow

---

## Future Improvements

- Implement transfer learning (ResNet)
- Add confusion matrix visualization
- Improve accuracy beyond 85%
- Deploy as a web application

---

## Author

K M Suhas  
Computer Science Student | Machine Learning Enthusiast
