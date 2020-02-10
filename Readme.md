# Image Classification using custom ResNet
The goal of this project is to build and train a model which is able to classify different fruits and vegetables.

## Tech used:
- TensorFlow 2.0.0
- Python 3.5.6

## Dataset:
- [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits) used for training and testing
- 82213 images of fruits and vegetables
- All images are RGB with dimensions 100 x 100 x 3 pixels
- Training set: 61488 images
- Test set: 20622 images
- Number of classes: 120
> Images of each class are taken from all sides (360 degrees) of the fruit or vegetable.

## Trained Models:
`model1.h5` has the following accuracy metrics:
  - Training accuracy = 99.21%
  - Validation accuracy = 92.50%
> `model1.h5` was trained for 20 epochs with a batch size of 32

## Instructions to run:
- Using `anaconda`:
  - Run `conda create --name <env_name> --file tf2.yml`
  - Run `conda activate <env_name>`
- Using `pip`:
  - Run `pip install -r requirements.txt`
- `mkdir datasets` in the same directory as `src`
- Download the [Fruits-360 Dataset](https://www.kaggle.com/moltean/fruits) into `datasets`
- `cd` to `src`
- Run `python main.py`