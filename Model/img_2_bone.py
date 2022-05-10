from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, Add, ReLU, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.activations import softmax
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, mse
from tensorflow.keras.optimizers import Adam
from tensorflow.math import multiply, reduce_mean, square


def create_img_2_bone(latent_dim = 64, dims = 128, kernal_size = 3):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,3)

    #image encoder input
    image_encoder_input = Input(shape=input_shape)


    #downsampling/encoder
    x1 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(image_encoder_input)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)

    x2 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)

    x3 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D((2, 2), padding='same')(x3)

    x4 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D((2, 2), padding='same')(x4)

    x5 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = MaxPooling2D((2, 2), padding='same')(x5)

    x5 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = Conv2D(32, (kernal_size, kernal_size), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = MaxPooling2D((2, 2), padding='same')(x5)


    x5 = Flatten()(x5)

    bottle_neck = Dense(latent_size)(x5)

     # first layer
    # flat = Flatten()(bottle_neck)
    # d1 = Dense(64)(flat)

    # block 1
    # x6 = Dense(64)(bottle_neck)
    x6 = BatchNormalization()(bottle_neck)
    x6 = ReLU()(x6)
    x6 = Dropout(.5)(x6)
    x6 = Dense(latent_size)(x6)
    x6 = BatchNormalization()(x6)
    x6 = ReLU()(x6)
    x6 = Dropout(.5)(x6)

    #skip connection
    skip_1 = Add()([bottle_neck,x6])

    # block 3
    x7 = Dense(latent_size)(skip_1)
    x7 = BatchNormalization()(x7)
    x7 = ReLU()(x7)
    x7 = Dropout(.5)(x7)
    x7 = Dense(latent_size)(x7)
    x7 = BatchNormalization()(x7)
    x7 = ReLU()(x7)
    x7 = Dropout(.5)(x7)

    #skip connection
    skip_2 = Add()([skip_1,x7])

    #output 
    output = Dense(52*3)(skip_2)

    output = Reshape((52,3))(output)

    # instantiate bone decoder model
    model = Model(image_encoder_input, output, name='model')

    model.summary()

    model.compile(optimizer='adam',loss=MeanSquaredError())

    return model


                 