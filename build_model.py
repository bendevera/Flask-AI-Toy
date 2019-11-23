from model import NLPModel, DiamondPredictor
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


def build_model():
    # builds sentiment classifier and vectorizer
    model = NLPModel()
    train_data_dir = 'lib/data/train.tsv'
    with open(train_data_dir) as f:
        data = pd.read_csv(f, sep='\t')

    pos_neg = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]
    pos_neg['Binary'] = pos_neg.apply(
        lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)

    model.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    X = model.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete')

    y = pos_neg.loc[:, 'Binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.train(X_train, y_train)

    model.pickle_clf()
    model.pickle_vectorizer(
    print('Sentiment Classifier Built')

    # builds diamond price predictor
    model_two = DiamondPredictor() 
    df = sns.load_dataset('diamonds')
    train, test = train_test_split(df.copy(), random_state=0)
    cut_ranks = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    train.cut = train.cut.map(cut_ranks)
    test.cut = test.cut.map(cut_ranks)
    features = ['carat', 'cut']
    target = 'price'
    model_two.train(train[features], train[target])
    model_two.pickle_model()
    print('Diamond Regressor Built')




if __name__ == "__main__":
    build_model()
