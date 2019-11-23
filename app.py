from flask import Flask, render_template
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel, DiamondPredictor
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

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

        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

class PredictDiamond(Resource):
    def get(self):
        args = parser.parse_args() 
        print(args)
        carat = float(args['carat'])
        cut = float(args['cut'])

        prediction = '~$' + str(round(float(model_two.predict([[carat, cut]])[0]), 2))
        print(prediction)
        return {'prediction': prediction}

# Setup the API Resources routing here
api.add_resource(PredictSentiment, '/api/sentiment')
api.add_resource(PredictDiamond, '/api/diamond')

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
