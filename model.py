from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from keras.models import load_model
import cv2
import numpy as np


class NLPModel(object):

    def __init__(self):
        """Simple NLP
        Attributes:
            clf: sklearn classifier model
            vectorizor: TFIDF vectorizer or similar
        """
        self.clf = MultinomialNB()
        # self.vectorizer = TfidfVectorizer(tokenizer=spacy_tok)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fits a TFIDF vectorizer to the text
        """
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        """Transform the text data to a sparse TFIDF matrix
        """
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):
        """Trains the classifier to associate the label with the sparse matrix
        """
        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        self.clf.fit(X, y)

    def predict_proba(self, X):
        """Returns probability for the binary class '1' in a numpy array
        """
        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):
        """Returns the predicted class in an array
        """
        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='lib/models/TFIDFVectorizer.pkl'):
        """Saves the trained vectorizer for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='lib/models/SentimentClassifier.pkl'):
        """Saves the trained classifier for future use.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))


class DiamondPredictor(object):
    def __init__(self):
        self.predictor = LinearRegression()

    def train(self, x, y): 
        self.predictor.fit(x, y)

    def predict(self, x):
        y_pred = self.predictor.predict(x)
        return y_pred 
    
    def pickle_model(self, path='lib/models/DiamondPredictor.sav'):
        with open(path, 'wb') as f:
            pickle.dump(self.predictor, f)
            print('Pickeled model at {}'.format(path))


class CatDogPredictor(object):
    def __init__(self):
        cat_dog_clf_path = 'lib/models/cats_and_dogs_v1.h5'
        self.predictor = load_model(cat_dog_clf_path)
        # cat_dog_model.compile(loss='binary_crossentropy',
        #     optimizer='rmsprop',
        #     metrics=['accuracy'])
    
    def predict(self, img_path):
        curr_img = cv2.imread(img_path)
        curr_img = cv2.resize(curr_img,(150,150))
        curr_img = np.reshape(curr_img,[1,150,150,3])
        prediction = self.predictor.predict(curr_img)
        return prediction 
