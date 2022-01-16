# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import pandas as pd
import pickle as pkl
from ml.data import process_data
from ml import model

import config

# Add code to load in the data.
data = pd.read_csv(config.PATH_DATA)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=config.TEST_SIZE)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=config.cat_features, label=config.TARGET, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=config.cat_features, label=config.TARGET, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
clf = model.train_model(X_train, y_train)

with open(config.MODEL_PATH, "wb") as file:
    pkl.dump([encoder, lb, clf], file)

train_preds = model.inference(clf, X_train)
test_preds = model.inference(clf, X_test)

precision, recall, fbeta = model.compute_model_metrics(y_test, test_preds)

print(f"Precicion: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {fbeta:.2f}")

model.compute_metrics_by_slice(
    clf=clf,
    encoder=encoder,
    lb=lb,
    df=test,
    target=config.TARGET,
    cat_columns=config.cat_features,
    output_path=config.METRICS_PATH,
)