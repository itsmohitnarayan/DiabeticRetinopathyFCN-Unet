import cv2
import os
import numpy as np
from skimage.filters import threshold_otsu
from keras.src.models import Model
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import tensorflow as tf

# Path to the dataset
dataset_path = r'D:\2024\DiabeticRetinopathyFCN-Unet\data'

# Step 1: Load and Preprocess the Images (CLAHE and Median Filter)
def load_and_preprocess_image(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    
    # Resize the image to a smaller dimension
    image = cv2.resize(image, (256, 256))  # Adjust as necessary
    
    # Apply median filter for noise reduction
    median_filtered = cv2.medianBlur(image, 5)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(median_filtered)
    
    return enhanced_image

# Step 2: Otsu Thresholding Segmentation
def apply_otsu_thresholding(image):
    # Use single Otsu thresholding
    threshold_value = threshold_otsu(image)
    segmented_image = image > threshold_value
    return segmented_image.astype(np.uint8)

# Step 3: Build a Pretrained U-Net Model
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    # Additional convolution layers to preserve spatial dimensions
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)

    # Expanding path
    u1 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)

    u2 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)

    u3 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Step 4: Compile and Evaluate the Model on the Segmented Images
def compile_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
    
    # Evaluate the model on test data
    score = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {score[1]*100:.2f}%")

# Main function to process multiple images
def process_images(dataset_path, limit=500):
    processed_images = []
    segmented_images = []
    
    # Loop through the first 500 images in the dataset
    for i, filename in enumerate(os.listdir(dataset_path)):
        if i >= limit:  # Process only the first 500 images
            break
        image_path = os.path.join(dataset_path, filename)
        
        # Step 1: Preprocess image (CLAHE and Median Filter)
        enhanced_image = load_and_preprocess_image(image_path)
        processed_images.append(enhanced_image)
        cv2.imwrite(f'D:/2024/DiabeticRetinopathyFCN-Unet/enhanced/enhanced_{i}.png', enhanced_image)  # Save preprocessed image
        
        # Step 2: Apply Otsu thresholding for segmentation
        segmented_image = apply_otsu_thresholding(enhanced_image)
        segmented_images.append(segmented_image)
        cv2.imwrite(f'D:/2024/DiabeticRetinopathyFCN-Unet/segmented/segmented_{i}.png', segmented_image)  # Save segmented image
        
    return np.array(processed_images), np.array(segmented_images)

# Train the model on the dataset
def train_on_dataset(dataset_path, limit=500):
    # Step 1 & 2: Process images and segmentation
    X_train, y_train = process_images(dataset_path, limit=limit)
    
    # Expand dimensions for the model input
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    # Step 3: Load U-Net model
    model = unet_model(input_size=(X_train.shape[1], X_train.shape[2], 1))
    
    # Split data for training and testing
    split_index = int(0.8 * len(X_train))  # 80% for training, 20% for testing
    X_test = X_train[split_index:]
    y_test = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    
    # Step 5: Compile, Train, and Evaluate Model
    compile_and_evaluate_model(model, X_train, y_train, X_test, y_test)

# Start the process
train_on_dataset(dataset_path)
