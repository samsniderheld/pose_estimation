from flask import Flask, current_app, request, send_file, Response
import json
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from scipy.spatial import distance
import scipy.misc
from keras.preprocessing import image
from Model.pose_detection_model import create_pose_detector
from Model.bone_auto_encoder import create_bone_auto_encoder



img_dim = 128

# encoder, bone_decoder, auto_encoder = create_bone_auto_encoder(
#         dims=img_dim , latent_dim = 128)

# auto_encoder.load_weights('Saved_Models/bone_auto_encoder_model.h5')

pose_detector = create_pose_detector()

pose_detector.load_weights('Saved_Models/pose_detector_model.h5')


app = Flask(__name__)
@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.form['img']
    except Exception:
        return jsonify(status_code='400', msg='Bad Request'), 400


    bone_values = data[:-11].split(",")

    print(len(bone_values))

    np_bone_values = np.array(bone_values).astype(np.float)

    pose = np_bone_values.reshape((1,17,2))

    prediction = pose_detector(pose)

    response = {"bones": prediction[0].numpy().flatten().tolist()}

    return json.dumps(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)