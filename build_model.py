from model import NLPModel, DiamondPredictor
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import os, shutil
from keras import layers 
from keras import models 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


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
    model.pickle_vectorizer()
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

def prep_cat_dog_images():
    print('Starting to prep images for cat / dog training.')

    # data link https://www.kaggle.com/c/dogs-vs-cats/data
    # since this is only being ran once I have data in Downloads and not in my repo
    original_dataset_dir = "/Users/bendevera/Downloads/dogs-vs-cats/train"

    base_dir = "/Users/bendevera/Downloads/dogs-vs-cats-small"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)
    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))

def build_cat_dog_model():
    print('Building Cat Dog Model')
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'])

    base_dir = "/Users/bendevera/Downloads/dogs-vs-cats-small"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)
    # train_datagen = ImageDataGenerator(rescale=1./255)
    # test_datagen = ImageDataGenerator(rescale=1./255)

    # train_generator = train_datagen.flow_from_directory(
    #     train_dir,
    #     target_size=(150,150),
    #     batch_size=20,
    #     class_mode='binary')

    # validation_generator = test_datagen.flow_from_directory(
    #     validation_dir,
    #     target_size=(150,150),
    #     batch_size=20,
    #     class_mode='binary')
    
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=100,
    #     epochs=30,
    #     validation_data=validation_generator,
    #     validation_steps=50)
    
    model.save('lib/models/cats_and_dogs_v1.h5')
    print('Cat Dog Model built and saved.')

if __name__ == "__main__":
    # build_model()
    # prep_cat_dog_images()
    build_cat_dog_model()
