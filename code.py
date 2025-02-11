import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained EfficientNetB0 model (better than MobileNetV2)
model = EfficientNetB0(weights="imagenet")

# Define ImageNet categories for dogs and cats
DOG_CLASSES = [
    "beagle", "pug", "golden_retriever", "labrador_retriever", "bulldog",
    "german_shepherd", "siberian_husky", "boxer", "dalmatian", "rottweiler",
    "doberman", "great_dane", "chihuahua", "pomeranian", "shih-tzu"
]
CAT_CLASSES = [
    "tabby", "tiger_cat", "persian_cat", "siamese_cat", "egyptian_cat"
]

def load_and_preprocess_image(img_path):
    """Load an image, convert it to RGB, resize it, and preprocess it for EfficientNetB0."""
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize for EfficientNetB0
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand batch dimension
    img_array = preprocess_input(img_array)
    
    return img_array, img

def classify_image(img_path):
    """Classifies an image as a dog or a cat using EfficientNetB0."""
    try:
        img_array, img = load_and_preprocess_image(img_path)
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions
        
        # Extract class labels and confidence scores
        best_label = decoded_preds[0][1].lower()  # Best predicted class
        confidence = decoded_preds[0][2] * 100  # Confidence score
        
        # Check if it belongs to dog or cat categories
        if any(dog in best_label for dog in DOG_CLASSES):
            prediction = "Dog"
        elif any(cat in best_label for cat in CAT_CLASSES):
            prediction = "Cat"
        else:
            prediction = "Neither Dog Nor Cat"

        # Display Image and Prediction
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Prediction: {prediction} ({confidence:.2f}%)")
        plt.show()

        return prediction, confidence, best_label

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the image path: ")  # Get image path from user
    result, confidence, label = classify_image(image_path)
    if result:
        print(f"Predicted: {result} ({confidence:.2f}%) - Class: {label}")

