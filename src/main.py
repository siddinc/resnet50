from glob import glob
import constants
import model_fn
import utility_fn


def main():
    utility_fn.mkdir('../datasets/fruits-360-small')
    utility_fn.mkdir(constants.TRAIN_PATH_TO)
    utility_fn.mkdir(constants.TEST_PATH_TO)

    for c in constants.CLASSES:
        utility_fn.link(constants.TRAIN_PATH_FROM + '/' + c,
                        constants.TRAIN_PATH_TO + '/' + c)
        utility_fn.link(constants.TEST_PATH_FROM + '/' + c,
                        constants.TEST_PATH_TO + '/' + c)

    train_image_files = glob(constants.TRAIN_PATH + '/*/*.jp*g')
    test_image_files = glob(constants.TEST_PATH + '/*/*.jp*g')

    print('Found {} training images and {} testing images'.format(
        len(train_image_files), len(test_image_files)))


if __name__ == '__main__':
    main()
