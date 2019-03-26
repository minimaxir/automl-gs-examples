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
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb


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
    age_encoder = MinMaxScaler()
    age_encoder_attrs = ['min_', 'scale_']
    age_encoder.fit(df['Age'].values.reshape(-1, 1))

    age_encoder_dict = {attr: getattr(age_encoder, attr).tolist()
                        for attr in age_encoder_attrs}

    with open(os.path.join('encoders', 'age_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(age_encoder_dict, outfile, ensure_ascii=False)

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
    fare_encoder = MinMaxScaler()
    fare_encoder_attrs = ['min_', 'scale_']
    fare_encoder.fit(df['Fare'].values.reshape(-1, 1))

    fare_encoder_dict = {attr: getattr(fare_encoder, attr).tolist()
                         for attr in fare_encoder_attrs}

    with open(os.path.join('encoders', 'fare_encoder.json'),
              'w', encoding='utf8') as outfile:
        json.dump(fare_encoder_dict, outfile, ensure_ascii=False)

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
    age_encoder = MinMaxScaler()
    age_encoder_attrs = ['min_', 'scale_']

    with open(os.path.join('encoders', 'age_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        age_attrs = json.load(infile)

    for attr, value in age_attrs.items():
        setattr(age_encoder, attr, value)
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
    fare_encoder = MinMaxScaler()
    fare_encoder_attrs = ['min_', 'scale_']

    with open(os.path.join('encoders', 'fare_encoder.json'),
              'r', encoding='utf8', errors='ignore') as infile:
        fare_attrs = json.load(infile)

    for attr, value in fare_attrs.items():
        setattr(fare_encoder, attr, value)
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
    age_enc = df['Age'].values.reshape(-1, 1)
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
    fare_enc = df['Fare'].values.reshape(-1, 1)
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

    data_enc = xgb.DMatrix(np.hstack(data_enc))

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

    X, y_enc = process_data(df, encoders)
    X = np.hstack(X)
    y = df['Survived'].values

    split = StratifiedShuffleSplit(
        n_splits=1, train_size=args.split, test_size=None, random_state=123)

    for train_indices, val_indices in split.split(np.zeros(y.shape[0]), y):
        train = xgb.DMatrix(X[train_indices, ], y[train_indices, ])
        val = xgb.DMatrix(X[val_indices, ], y[val_indices, ])

    params = {
        'eta': 0.1,
        'max_depth': 7,
        'gamma': 10,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 0.8,
        'max_bin': 256,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'silent': 1
    }

    f = open(os.path.join('metadata', 'results.csv'), 'w')
    w = csv.writer(f)
    w.writerow(['epoch', 'time_completed'] + ['log_loss',
                                              'accuracy', 'auc', 'precision', 'recall', 'f1'])

    y_true = y_enc[val_indices, ]
    for epoch in range(args.epochs):
        model = xgb.train(params, train, 1,
                          xgb_model=model if epoch > 0 else None)
        y_pred = model.predict(val)

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
        w.writerow([epoch+1, time_completed] + metrics)

        if args.context == 'automl-gs':
            sys.stdout.flush()
            print("\nEPOCH_END")

    f.close()
    model.save_model('model.bin')
