from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Lambda, Add, ReLU, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

def create_img_2_bone(latent_dim = 64, dims = 128, kernal_size = 3):
    #define model
    latent_size = latent_dim

    original_dims = dims * dims

    input_shape = (dims,dims,1)

    #image encoder input
    image_encoder_input = Input(shape=input_shape)
    #bone encoder input
    bone_encoder_input = Input(shape=(52,3))

    #bone encoder input

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
    x6 = Dense(64)(x6)
    x6 = BatchNormalization()(x6)
    x6 = ReLU()(x6)
    x6 = Dropout(.5)(x6)

    #skip connection
    skip_1 = Add()([bottle_neck,x6])

    # block 3
    x7 = Dense(64)(skip_1)
    x7 = BatchNormalization()(x7)
    x7 = ReLU()(x7)
    x7 = Dropout(.5)(x7)
    x7 = Dense(64)(x7)
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

    #define losses    
    opt = Adam(learning_rate=0.0005)

    weighted_matrix = tf.linespace(1.0,.1,52)

    weighted_output = Multiply()([weighted_matrix,output])

    bone_reconstruction_loss = mse(K.flatten(bone_encoder_input), K.flatten(weighted_output))
    
    model.add_loss(bone_reconstruction_loss)

    # model.compile(optimizer=opt,loss=MeanSquaredError())
    model.compile(optimizer=opt)

    return model


                 