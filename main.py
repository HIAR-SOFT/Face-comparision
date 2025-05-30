import cv2
import numpy as np
import os

def image_to_vector(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    
    image_resized = cv2.resize(image, (50, 50))  # Resize to fixed shape
    vector = image_resized.flatten() / 255.0     # Normalize pixel values
    return vector

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Paths to two face images
img1_path = os.path.join("images", "person1.jpg")
img2_path = os.path.join("images", "person2.jpg")

# Convert images to vectors
vec1 = image_to_vector(img1_path)
vec2 = image_to_vector(img2_path)

# Calculate distance
distance = euclidean_distance(vec1, vec2)
print(f"Euclidean Distance: {distance:.4f}")

# Threshold (arbitrary): smaller = more similar
threshold = 5.0
if distance < threshold:
    print("✅ Faces are similar!")
else:
    print("❌ Faces are different.")
