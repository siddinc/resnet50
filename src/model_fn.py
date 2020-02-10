from datetime import datetime
from tensorflow.keras.layers import (
    Input, Add, Dense, Activation,
    BatchNormalization, Flatten,
    Conv2D, ZeroPadding2D,
    AveragePooling2D, MaxPooling2D,
)
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import constants


def identity_block(x, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    x_shortcut = x

    # First component of main path
    x = Conv2D(F1, (1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(F2, (f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Add shortcut path to main path
    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)
    return x


def convolutional_block(x, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    x_shortcut = x

    # First component of main path
    x = Conv2D(F1, (1, 1), strides=(s, s), padding='valid',
               name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(F2, (f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    # Shortcut path
    x_shortcut = Conv2D(F3, (1, 1), strides=(
        s, s), padding='valid', name=conv_name_base + '1')(x)
    x_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(x)

    # Add shortcut path to main path
    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)
    return x


def create_model(input_shape, classes):
    # Input
    i = Input(shape=input_shape)

    # ZeroPadding
    x = ZeroPadding2D((3, 3))(i)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, 3, [64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    x = convolutional_block(x, 3, [128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = convolutional_block(x, 3, [256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = convolutional_block(x, 3, [512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # AveragePooling
    x = AveragePooling2D((2, 2), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    model = Model(i, x)
    return model


def compile_model(model, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def data_augmentation(batch_size, image_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
    )

    train_generator = data_generator.flow_from_directory(
        constants.TRAIN_PATH,
        target_size=image_size,
        shuffle=True,
        batch_size=batch_size
    )

    validation_generator = data_generator.flow_from_directory(
        constants.TEST_PATH,
        target_size=image_size,
        shuffle=True,
        batch_size=batch_size,
    )
    return (train_generator, validation_generator)


def train_model(model, generators, no_of_epochs, batch_size, len_train_image_files, len_test_image_files):
    r = model.fit_generator(
        generators[0],
        validation_data=generators[1],
        epochs=no_of_epochs,
        steps_per_epoch=len_train_image_files // batch_size,
        validation_steps=len_test_image_files // batch_size,
    )
    return r


def save_model(model):
    now = datetime.now()
    model_name_suffix = now.strftime('%d/%m/%Y-%H:%M:%S')
    save_model(model, constants.SAVE_MODEL_PATH +
               '/model${}'.format(model_name_suffix))


def load_saved_model(model_name):
    loaded_model = load_model(
        constants.LOAD_MODEL_PATH + '/{}'.format(model_name))
    return loaded_model
