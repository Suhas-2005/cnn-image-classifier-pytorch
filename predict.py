import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CIFAR10CNN, CLASSES
import numpy as np
import sys
import os

class CIFAR10Predictor:
    def __init__(self, model_path='cifar10_best.pth'):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = CIFAR10CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def predict_image(self, image_path):
        """Predict the class of an image from local file."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.exp(outputs)
                
                # Get top predictions and their probabilities
                top_prob, top_class = torch.topk(probabilities, 5)
                
                # Convert to numpy for easier handling
                top_prob = top_prob.cpu().numpy()[0]
                top_class = top_class.cpu().numpy()[0]
                
                # Format results
                predictions = []
                for prob, class_idx in zip(top_prob, top_class):
                    predictions.append({
                        'class': CLASSES[class_idx],
                        'confidence': float(prob)
                    })
                
                return predictions
                
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

def print_usage():
    print("\nUsage:")
    print("1. To test with your own image:")
    print("   python predict.py <path_to_your_image>")
    print("\nExample:")
    print("   python predict.py C:\\Users\\YourName\\Pictures\\cat.jpg")
    print("\nThe model can recognize these categories:")
    print("   " + ", ".join(CLASSES))

def main():
    if len(sys.argv) < 2:
        print_usage()
        return
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print_usage()
        return
    
    # Example usage
    predictor = CIFAR10Predictor()
    
    try:
        print(f"\nAnalyzing image: {image_path}")
        predictions = predictor.predict_image(image_path)
        
        # Get the top prediction
        top_prediction = predictions[0]
        
        print("\n" + "="*50)
        print(f"This image is identified as: {top_prediction['class'].upper()}")
        print(f"Confidence: {top_prediction['confidence']*100:.1f}%")
        print("="*50)
        
        # Show other possibilities if confidence is not very high
        if top_prediction['confidence'] < 0.8:  # if less than 80% confident
            print("\nOther possibilities:")
            for i, pred in enumerate(predictions[1:], 2):
                print(f"{i}. {pred['class']} ({pred['confidence']*100:.1f}%)")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 