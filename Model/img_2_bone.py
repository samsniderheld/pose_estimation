from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError

def create_img_2_bone(latent_dim = 10, dims = 128, kernal_size = 3):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,1)

    #image encoder input
    image_encoder_input = Input(shape=input_shape)

    #bone encoder input

    #downsampling/encoder
    x1 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(image_encoder_input)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)

    x2 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2), padding='same')(x2)

    x3 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D((2, 2), padding='same')(x3)

    x4 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D((2, 2), padding='same')(x4)

    x5 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Conv2D(64, (kernal_size, kernal_size), activation='relu', padding='same')(x5)
    x5 = BatchNormalization()(x5)
    x5 = MaxPooling2D((2, 2), padding='same')(x5)


    x5 = Flatten()(x5)

    bottle_neck = Dense(latent_size)(x5)



    x6 = Dense(13*latent_size)(bottle_neck)

    x6 = Reshape((1,13, latent_size))(x6)
    x6 = Conv2DTranspose(256, (kernal_size, kernal_size), strides = (1,2), activation='relu', padding='same')(x6)
    x6 = BatchNormalization()(x6)

    x7 = Conv2DTranspose(256, (kernal_size, kernal_size), strides = (1,1), activation='relu', padding='same')(x6)
    x7 = BatchNormalization()(x7)

    x8 = Conv2DTranspose(128, (kernal_size, kernal_size), strides = (1,2), activation='relu', padding='same')(x7)
    x8 = BatchNormalization()(x8)

    x9 = Conv2DTranspose(128, (kernal_size, kernal_size), strides = (1,1), activation='relu', padding='same')(x8)
    x9 = BatchNormalization()(x9)

    decoder_output = Conv2DTranspose(6, (kernal_size, kernal_size), strides = (1,1), activation='sigmoid', padding='same')(x9)

    # instantiate bone decoder model
    unet_model = Model(image_encoder_input, decoder_output, name='unet_model')

    unet_model.summary()

    #define losses    
    unet_model.add_loss(bone_reconstruction_loss)

    unet_model.compile(optimizer='adam',loss=MeanSquaredError())

    return unet_model


                 