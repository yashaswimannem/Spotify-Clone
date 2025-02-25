from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle

# Load the pre-trained model
model = VGG16(weights='imagenet', include_top=False)

# Load and preprocess the image
img_path = 'image.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))  # Resize the image to 224x224 pixels
img_array = image.img_to_array(img)  # Convert the image to an array
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
img_array = preprocess_input(img_array)  # Preprocess the image for VGG16

# Extract features
features = model.predict(img_array)

# Flatten the features to create a feature vector
feature_vector = features.flatten()

# Print the shape of the feature vector
print("Feature vector shape:", feature_vector.shape)

# Optionally, save the feature vector to a file
with open('feature_vector.pkl', 'wb') as f:
    pickle.dump(feature_vector, f)
    print("Feature vector saved to feature_vector.pkl")

# Optionally, use the feature vector for further tasks (e.g., classification, clustering, etc.)
# For now, let's just print a small part of it
print("Feature vector (first 10 elements):", feature_vector[:10])
