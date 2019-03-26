from tensorflow.train import cosine_decay, AdamOptimizer
from tensorflow.contrib.opt import AdamWOptimizer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, CuDNNLSTM, GRU, CuDNNGRU, concatenate, Dense, BatchNormalization, Dropout, AlphaDropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
import csv
import sys
import warnings
from datetime import datetime
from math import floor
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.ERROR)


def build_model(encoders):
    """Builds and compiles the model from scratch.

    # Arguments
        encoders: dict of encoders (used to set size of text/categorical inputs)

    # Returns
        model: A compiled model which can be used to train or predict.
    """

    # Pclass
    input_pclass_size = len(encoders['pclass_encoder'].classes_)
    input_pclass = Input(shape=(
        input_pclass_size if input_pclass_size != 2 else 1,), name="input_pclass")

    # Sex
    input_sex_size = len(encoders['sex_encoder'].classes_)
    input_sex = Input(
        shape=(input_sex_size if input_sex_size != 2 else 1,), name="input_sex")

    # Age
    input_age = Input(shape=(10,), name="input_age")

    # Siblings/Spouses Aboard
    input_siblings_spouses_aboard_size = len(
        encoders['siblings_spouses_aboard_encoder'].classes_)
    input_siblings_spouses_aboard = Input(shape=(
        input_siblings_spouses_aboard_size if input_siblings_spouses_aboard_size != 2 else 1,), name="input_siblings_spouses_aboard")

    # Parents/Children Aboard
    input_parents_children_aboard_size = len(
        encoders['parents_children_aboard_encoder'].classes_)
    input_parents_children_aboard = Input(shape=(
        input_parents_children_aboard_size if input_parents_children_aboard_size != 2 else 1,), name="input_parents_children_aboard")

    # Fare
    input_fare = Input(shape=(10,), name="input_fare")

    # Combine all the inputs into a single layer
    concat = concatenate([
        input_pclass,
        input_sex,
        input_age,
        input_siblings_spouses_aboard,
        input_parents_children_aboard,
        input_fare
    ], name="concat")

    # Multilayer Perceptron (MLP) to find interactions between all inputs
    hidden = Dense(256, activation="relu", name="hidden_1",
                   kernel_regularizer=l2(1e-3))(concat)
    hidden = BatchNormalization(name="bn_1")(hidden)
    hidden = Dropout(0.0, name="dropout_1")(hidden)

    for i in range(2-1):
        hidden = Dense(64, activation="relu", name="hidden_{}".format(
            i+2), kernel_regularizer=l2(1e-3))(hidden)
        hidden = BatchNormalization(name="bn_{}".format(i+2))(hidden)
        hidden = Dropout(0.0, name="dropout_{}".format(i+2))(hidden)

    output = Dense(1, activation="sigmoid", name="output",
                   kernel_regularizer=None)(hidden)

    # Build and compile the model.
    model = Model(inputs=[
        input_pclass,
        input_sex,
        input_age,
        input_siblings_spouses_aboard,
        input_parents_children_aboard,
        input_fare
    ],
        outputs=[output])
    model.compile(loss="binary_crossentropy",
                  optimizer=AdamWOptimizer(learning_rate=0.1,
                                           weight_decay=0.05))

    return model


def build_encoders(df):
    """Builds encoders for fields to be used when
    processing data for the model.

    All encoder specifications are stored in locally
    in /encoders as .json files.

    # Arguments
        df: A pandas DataFrame containing the data.
    """

    # Pclass
    pclass_tf = df['Pclass'].values
    pclass_encoder = LabelBinarizer()
    pclass_encoder.fit(pclass_tf)

    with open(os.path.join('encoders', 'pclass_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(pclass_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Sex
    sex_tf = df['Sex'].values
    sex_encoder = LabelBinarizer()
    sex_encoder.fit(sex_tf)

    with open(os.path.join('encoders', 'sex_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(sex_encoder.classes_.tolist(), outfile, ensure_ascii=False)

    # Age
    age_enc = df['Age']
    age_bins = age_enc.quantile(np.linspace(0, 1, 10+1))

    with open(os.path.join('encoders', 'age_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(age_bins.tolist(), outfile, ensure_ascii=False)

    # Siblings/Spouses Aboard
    siblings_spouses_aboard_tf = df['Siblings/Spouses Aboard'].values
    siblings_spouses_aboard_encoder = LabelBinarizer()
    siblings_spouses_aboard_encoder.fit(siblings_spouses_aboard_tf)

    with open(os.path.join('encoders', 'siblings_spouses_aboard_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(siblings_spouses_aboard_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Parents/Children Aboard
    parents_children_aboard_tf = df['Parents/Children Aboard'].values
    parents_children_aboard_encoder = LabelBinarizer()
    parents_children_aboard_encoder.fit(parents_children_aboard_tf)

    with open(os.path.join('encoders', 'parents_children_aboard_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(parents_children_aboard_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)

    # Fare
    fare_enc = df['Fare']
    fare_bins = fare_enc.quantile(np.linspace(0, 1, 10+1))

    with open(os.path.join('encoders', 'fare_bins.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fare_bins.tolist(), outfile, ensure_ascii=False)

    # Target Field: Survived
    survived_encoder = LabelEncoder()
    survived_encoder.fit(df['Survived'].values)

    with open(os.path.join('encoders', 'survived_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(survived_encoder.classes_.tolist(),
                  outfile, ensure_ascii=False)


def load_encoders():
    """Loads the encoders built during `build_encoders`.

    # Returns
        encoders: A dict of encoder objects/specs.
    """

    encoders = {}

    # Pclass
    pclass_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'pclass_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        pclass_encoder.classes_ = json.load(infile)
    encoders['pclass_encoder'] = pclass_encoder

    # Sex
    sex_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'sex_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        sex_encoder.classes_ = json.load(infile)
    encoders['sex_encoder'] = sex_encoder

    # Age
    age_encoder = LabelBinarizer()
    age_encoder.classes_ = list(range(10))

    with open(os.path.join('encoders', 'age_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        age_bins = json.load(infile)
    encoders['age_bins'] = age_bins
    encoders['age_encoder'] = age_encoder

    # Siblings/Spouses Aboard
    siblings_spouses_aboard_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'siblings_spouses_aboard_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        siblings_spouses_aboard_encoder.classes_ = json.load(infile)
    encoders['siblings_spouses_aboard_encoder'] = siblings_spouses_aboard_encoder

    # Parents/Children Aboard
    parents_children_aboard_encoder = LabelBinarizer()

    with open(os.path.join('encoders', 'parents_children_aboard_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        parents_children_aboard_encoder.classes_ = json.load(infile)
    encoders['parents_children_aboard_encoder'] = parents_children_aboard_encoder

    # Fare
    fare_encoder = LabelBinarizer()
    fare_encoder.classes_ = list(range(10))

    with open(os.path.join('encoders', 'fare_bins.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fare_bins = json.load(infile)
    encoders['fare_bins'] = fare_bins
    encoders['fare_encoder'] = fare_encoder

    # Target Field: Survived
    survived_encoder = LabelEncoder()

    with open(os.path.join('encoders', 'survived_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        survived_encoder.classes_ = np.array(json.load(infile))
    encoders['survived_encoder'] = survived_encoder

    return encoders


def process_data(df, encoders, process_target=True):
    """Processes an input DataFrame into a format
    sutable for model prediction.

    This function loads the encoder specifications created in
    `build_encoders`.

    # Arguments
        df: a DataFrame containing the source data
        encoders: a dict of encoders to process the data.
        process_target: boolean to determine if the target should be encoded.

    # Returns
        A tuple: A list containing all the processed fields to be fed
        into the model, and the processed target field.
    """

    # Pclass
    pclass_enc = df['Pclass'].values
    pclass_enc = encoders['pclass_encoder'].transform(pclass_enc)

    # Sex
    sex_enc = df['Sex'].values
    sex_enc = encoders['sex_encoder'].transform(sex_enc)

    # Age
    age_enc = pd.cut(df['Age'].values, encoders['age_bins'],
                     labels=False, include_lowest=True)
    age_enc = encoders['age_encoder'].transform(age_enc)

    # Siblings/Spouses Aboard
    siblings_spouses_aboard_enc = df['Siblings/Spouses Aboard'].values
    siblings_spouses_aboard_enc = encoders['siblings_spouses_aboard_encoder'].transform(
        siblings_spouses_aboard_enc)

    # Parents/Children Aboard
    parents_children_aboard_enc = df['Parents/Children Aboard'].values
    parents_children_aboard_enc = encoders['parents_children_aboard_encoder'].transform(
        parents_children_aboard_enc)

    # Fare
    fare_enc = pd.cut(df['Fare'].values, encoders['fare_bins'],
                      labels=False, include_lowest=True)
    fare_enc = encoders['fare_encoder'].transform(fare_enc)

    data_enc = [pclass_enc,
                sex_enc,
                age_enc,
                siblings_spouses_aboard_enc,
                parents_children_aboard_enc,
                fare_enc
                ]

    if process_target:
        # Target Field: Survived
        survived_enc = df['Survived'].values

        survived_enc = encoders['survived_encoder'].transform(survived_enc)

        return (data_enc, survived_enc)

    return data_enc


def model_predict(df, model, encoders):
    """Generates predictions for a trained model.

    # Arguments
        df: A pandas DataFrame containing the source data.
        model: A compiled model.
        encoders: a dict of encoders to process the data.

    # Returns
        A numpy array of predictions.
    """

    data_enc = process_data(df, encoders, process_target=False)

    headers = ['probability']
    predictions = pd.DataFrame(model.predict(data_enc), columns=headers)

    return predictions


def model_train(df, encoders, args, model=None):
    """Trains a model, and saves the data locally.

    # Arguments
        df: A pandas DataFrame containing the source data.
        encoders: a dict of encoders to process the data.
        args: a dict of arguments passed through the command line
        model: A compiled model (for TensorFlow, None otherwise).
    """
    X, y = process_data(df, encoders)

    split = StratifiedShuffleSplit(
        n_splits=1, train_size=args.split, test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        X_train = [field[train_indices, ] for field in X]
        X_val = [field[val_indices, ] for field in X]
        y_train = y[train_indices, ]
        y_val = y[val_indices, ]

    meta = meta_callback(args, X_val, y_val)

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs,
              callbacks=[meta],
              batch_size=128)


class meta_callback(Callback):
    """Keras Callback used during model training to save current weights
    and metrics after each training epoch.

    Metrics metadata is saved in the /metadata folder.
    """

    def __init__(self, args, X_val, y_val):
        self.f = open(os.path.join('metadata', 'results.csv'), 'w')
        self.w = csv.writer(self.f)
        self.w.writerow(['epoch', 'time_completed'] + ['log_loss',
                                                       'accuracy', 'auc', 'precision', 'recall', 'f1'])
        self.in_automl = args.context == 'automl-gs'
        self.X_val = X_val
        self.y_val = y_val

    def on_train_end(self, logs={}):
        self.f.close()
        self.model.save_weights('model_weights.hdf5')

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.y_val
        y_pred = self.model.predict(self.X_val)

        y_pred_label = np.round(y_pred)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logloss = log_loss(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_label)
            precision = precision_score(y_true, y_pred_label, average='macro')
            recall = recall_score(y_true, y_pred_label, average='macro')
            f1 = f1_score(y_true, y_pred_label, average='macro')
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            auc_score = auc(fpr, tpr)

        metrics = [logloss,
                   acc,
                   auc_score,
                   precision,
                   recall,
                   f1]
        time_completed = "{:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())
        self.w.writerow([epoch+1, time_completed] + metrics)

        # Only run while using automl-gs, which tells it an epoch is finished
        # and data is recorded.
        if self.in_automl:
            sys.stdout.flush()
            print("\nEPOCH_END")
