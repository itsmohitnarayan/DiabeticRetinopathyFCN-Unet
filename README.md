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

## Results
The model achieves an accuracy of 99.39% on the test set. Below are some sample predictions:


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
//copilot prepare a readme for this project 

