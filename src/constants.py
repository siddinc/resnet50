import os


IMG_DIM = (100, 100, 3)
IMAGE_SIZE = [100, 100]
NO_OF_EPOCHS = 20
BATCH_SIZE = 32
TRAIN_PATH_FROM = os.path.abspath('../datasets/fruits-360/Training')
TEST_PATH_FROM = os.path.abspath('../datasets/fruits-360/Test')
TRAIN_PATH_TO = os.path.abspath('../datasets/fruits-360-small/Training')
TEST_PATH_TO = os.path.abspath('../datasets/fruits-360-small/Testing')
TRAIN_PATH = '../datasets/fruits-360-small/Training'
TEST_PATH = '../datasets/fruits-360-small/Testing'
SAVE_MODEL_PATH = os.path.abspath('../models')
LOAD_MODEL_PATH = os.path.abspath('../models')
CLASSES = [
    'Apple Golden 1',
    'Avocado',
    'Lemon',
    'Mango',
    'Kiwi',
    'Banana',
    'Strawberry',
    'Raspberry'
]
