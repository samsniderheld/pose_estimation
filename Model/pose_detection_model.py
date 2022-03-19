from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, ReLU, Dropout, Add, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError

def create_pose_detector():
    #define model

    model_input = Input(shape=(52,2))

    # first layer
    flat = Flatten()(model_input)
    d1 = Dense(1024)(flat)

    # block 1
    x1 = Dense(1024)(d1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Dropout(.5)(x1)
    x1 = Dense(1024)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Dropout(.5)(x1)

    #skip connection
    skip_1 = Add()([d1,x1])

    # block 3
    x2 = Dense(1024)(skip_1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Dropout(.5)(x2)
    x2 = Dense(1024)(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Dropout(.5)(x2)

    #skip connection
    skip_2 = Add()([skip_1,x2])

    #output 
    output = Dense(52*6, action="sigmoid")(skip_2)

    output = Reshape((52,6))(output)


    # instantiate the pose detector model
    pose_detector = Model (model_input,output, name="pose_detector")

    pose_detector.summary()

    pose_detector.compile(optimizer="adam", loss=MeanSquaredError())

    return pose_detector


                 