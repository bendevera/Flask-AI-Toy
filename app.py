from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel, DiamondPredictor, CatDogPredictor
from flask_cors import CORS
import tensorflow as tf
# from tensorflow.python.framework import ops
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
import cv2
import os
import time 

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)
UPLOAD_FOLDER = '/Users/bendevera/Desktop/development/data_science/Flask-AI-Toy/static/uploads'
print(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = NLPModel()
model_two = DiamondPredictor()

clf_path = 'lib/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'lib/models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

regressor_path = 'lib/models/DiamondPredictor.sav'
with open(regressor_path, 'rb') as f:
    model_two.predictor = pickle.load(f) 

global graph
graph = tf.get_default_graph()
sess = tf.Session()
# graph = ops.get_default_graph() 
print(graph)
# cat_dog_model = CatDogPredictor()
cat_dog_clf_path = 'lib/models/cats_and_dogs_v1.h5'
set_session(sess)
cat_dog_model = load_model(cat_dog_clf_path)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')
parser.add_argument('carat')
parser.add_argument('cut')


class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query'].replace("-", " ")
        print(user_query)
        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
        # Output either 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'
        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 2)
        response = {'success': True, 'message': str(confidence) + "% sure it's " + pred_text, 'sentiment': pred_text}
        print(response)
        return response

class PredictDiamond(Resource):
    def get(self):
        args = parser.parse_args() 
        print(args)
        carat = float(args['carat'])
        cut = float(args['cut'])

        prediction = '~$' + str(round(float(model_two.predict([[carat, cut]])[0]), 2))
        print(prediction)
        return {'success': True, 'message': prediction}

class CatAndDogPredictor(Resource):
    def get(self):
        # args = parser.parse_args()
        curr_img = 'lib/data/test_dog.jpg'
        curr_img = cv2.imread(curr_img)
        curr_img = cv2.resize(curr_img,(150,150))
        curr_img = np.reshape(curr_img,[1,150,150,3])
        with graph.as_default():
            set_session(sess)
            prediction = cat_dog_model.predict(curr_img)[0][0]
        print(prediction)
        if prediction == 1:
            prediction = 'Dog'
        else:
            prediction = 'Cat'
        print(prediction)
        return {'success': True, 'message': prediction}


# Setup the API Resources routing here
api.add_resource(PredictSentiment, '/api/sentiment')
api.add_resource(PredictDiamond, '/api/diamond')
# api.add_resource(CatAndDogPredictor, '/api/cat-and-dog')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/cat-and-dog', methods=["POST"])
def cat_and_dog():
    if 'file' not in request.files:
        return {'success': False, 'message': 'No file in post request'}
    file = request.files['file']
    if file.filename == '':
        return {'success': False, 'message': 'No file in post request'}
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        time.sleep(.1)
        curr_img = cv2.imread(filename)
        curr_img = cv2.resize(curr_img,(150,150))
        curr_img = np.reshape(curr_img,[1,150,150,3])
        with graph.as_default():
            set_session(sess)
            prediction = cat_dog_model.predict(curr_img)[0][0]
        print(prediction)
        if prediction == 1:
            prediction = 'Dog'
        else:
            prediction = 'Cat'
        print(prediction)
        return {'success': True, 'message': prediction}



if __name__ == '__main__':
    app.run(debug=True)
