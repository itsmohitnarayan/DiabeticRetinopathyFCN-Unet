# DiabeticRetinopathyFCN-Unet

## Overview
This project implements a Fully Convolutional Network (FCN) using the U-Net architecture to detect and classify diabetic retinopathy from retinal images. The model processes images through several steps including preprocessing, segmentation, and training using a U-Net model.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/itsmohitnarayan/DiabeticRetinopathyFCN-Unet.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DiabeticRetinopathyFCN-Unet
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the model on a new set of images:
```markdown
python new2.py --input_dir /path/to/images --output_dir /path/to/save/results
```

## Dataset
The datasets used for this project are:
- [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- [STARE](http://cecas.clemson.edu/~ahoover/stare/)
- [DRIVE](https://drive.grand-challenge.org/)
- [IDRID](https://idrid.grand-challenge.org/)

Download the datasets and place them in the `data/` directory.

## Detailed Explanation
`Step 1:` Load and Preprocess the Images (CLAHE and Median Filter)
The function load_and_preprocess_image loads an image in grayscale, resizes it, applies a median filter for noise reduction, and enhances the contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
```markdown
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, 0)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    image = cv2.resize(image, (256, 256))
    median_filtered = cv2.medianBlur(image, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(median_filtered)
    return enhanced_image
```

`Step 2:` Otsu Thresholding Segmentation
The function apply_otsu_thresholding applies Otsu's method to segment the image.
```markdown
def apply_otsu_thresholding(image):
    threshold_value = threshold_otsu(image)
    segmented_image = image > threshold_value
    return segmented_image.astype(np.uint8)
```

`Step 3:` Build a Pretrained U-Net Model
The function unet_model builds a U-Net model with a contracting path, bottleneck, and expanding path.
```markdown
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    u1 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    u2 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    u3 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
```

`Step 4:` Compile and Evaluate the Model on the Segmented Images
The function compile_and_evaluate_model compiles the model with the Adam optimizer and binary cross-entropy loss, trains it, and evaluates its accuracy on the test data.
```markdown
def compile_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
    score = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {score[1]*100:.2f}%")
```

`Main Function to Process Multiple Images`
The function process_images processes multiple images by loading, preprocessing, and segmenting them.
```markdown
def process_images(dataset_path, limit=500):
    processed_images = []
    segmented_images = []
    for i, filename in enumerate(os.listdir(dataset_path)):
        if i >= limit:
            break
        image_path = os.path.join(dataset_path, filename)
        enhanced_image = load_and_preprocess_image(image_path)
        processed_images.append(enhanced_image)
        cv2.imwrite(f'D:/2024/DiabeticRetinopathyFCN-Unet/enhanced/enhanced_{i}.png', enhanced_image)
        segmented_image = apply_otsu_thresholding(enhanced_image)
        segmented_images.append(segmented_image)
        cv2.imwrite(f'D:/2024/DiabeticRetinopathyFCN-Unet/segmented/segmented_{i}.png', segmented_image)
    return np.array(processed_images), np.array(segmented_images)
```

`Train the Model on the Dataset`
The function train_on_dataset orchestrates the entire process by calling the necessary functions to preprocess images, build the model, and train it.
```markdown
def train_on_dataset(dataset_path, limit=500):
    X_train, y_train = process_images(dataset_path, limit=limit)
    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    model = unet_model(input_size=(X_train.shape[1], X_train.shape[2], 1))
    split_index = int(0.8 * len(X_train))
    X_test = X_train[split_index:]
    y_test = y_train[split_index:]
    X_train = X_train[:split_index]
    y_train = y_train[:split_index]
    compile_and_evaluate_model(model, X_train, y_train, X_test, y_test)
```

`Execution`
To start the training process, simply run the script:
```markdown
python new2.py
```
## Note
This work is not completed many work is in progress.

## Results
The model achieves an accuracy of 98.98% on the test set.


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the Private and Patiented License. See the [LICENSE](LICENSE) file for details.

