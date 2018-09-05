from flask_restful import Resource, Api
import keras
from flask import Flask, request, render_template
from sklearn.externals import joblib
from flask_cors import CORS
import numpy as np
from scipy import misc
import tensorflow as tf
from keras.models import load_model
import os
import pandas as pd
import cv2 as cv

app = Flask(__name__)
api= Api(app)
cors = CORS(app)
#model.h5 for ANN and cnn_model for CNN
model = load_model('cnn_model.h5')
graph = tf.get_default_graph()

UPLOAD_FOLDER = "/home/uttam/PycharmProjects/ML"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
class Upload(Resource):

    def post(self):
        file = request.files['image']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'])+"predict.jpg")


        return {"status": "ok"}

class Predict(Resource):

    def get(self):
        datas = []
        label_data = pd.read_csv('labels.csv',delimiter=',')
        test_image = cv.imread(os.path.join(app.config['UPLOAD_FOLDER'])+"predict.jpg", 0)
        image_array = cv.resize(test_image, (36, 36))
        (thresh, image_array) = cv.threshold(image_array, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        # X_test = image_array
        X_test = image_array.astype('float32')
        X_test /= 255
        datas.append(X_test)
        # for ANN
        # X_test = np.array(datas)
        # for CNN
        X_test = np.array(datas).reshape(-1,36,36,1)
        with graph.as_default():
            prediction = model.predict_classes(X_test)
            prediction_prob = model.predict_proba(X_test)
        label = int(np.squeeze(prediction))
        prob = np.squeeze(prediction_prob)
        probability = np.amax(prob)
        if label == '10': label = '0'
        return {"prediction": str(label_data.iloc[label, :].values[0]), "probability": float(probability)}


api.add_resource(Upload, '/upload')
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run()