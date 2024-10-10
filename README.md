# DiabeticRetinopathyFCN-Unet

## Overview
This project implements a Fully Convolutional Network (FCN) using the U-Net architecture to detect and classify diabetic retinopathy from retinal images.

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
To run the model on a new set of images:
```bash
python new2.py --input_dir /path/to/images --output_dir /path/to/save/results
```

## Dataset
The datasets used for this project are:
- [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- [STARE](http://cecas.clemson.edu/~ahoover/stare/)
- [DRIVE](https://drive.grand-challenge.org/)
- [IDRID](https://idrid.grand-challenge.org/)

Download the datasets and place them in the `data/` directory.


## Results
The model achieves an accuracy of 99.39% on the test set. Below are some sample predictions:


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
//copilot prepare a readme for this project 

