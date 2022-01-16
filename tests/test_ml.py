"""Tests for the ML section."""

import os
import pickle as pkl

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, compute_metrics_by_slice
import starter.config as config


@pytest.fixture()
def input_df():
    df = pd.read_csv(config.DATA_PATH)
    train, test = train_test_split(df, test_size=config.TEST_SIZE)
    return train, test


def test_process_data(input_df):
    train, test = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=config.cat_features, label=config.TARGET, training=True
    )

    # Test the number of rows for training dataset
    assert len(X_train) == len(y_train)

    X_test, y_test, encoder_test, lb_test = process_data(
        X=train, categorical_features=config.cat_features, label=config.TARGET, training=False, encoder=encoder, lb=lb
    )

    # Test the number of generated features for train and test datasets
    assert X_train.shape[1] == X_test.shape[1]


def test_inference(input_df):
    train, test = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=config.cat_features, label=config.TARGET, training=True
    )

    clf = train_model(X_train, y_train)

    train_predictions = inference(clf, X_train)

    # All predictions should be less than 1
    assert all(train_predictions <= 1.0)

    # All predictions should be positive
    assert all(train_predictions >= 0)


def test_compute_metrics(input_df):

    train, test = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=config.cat_features, label=config.TARGET, training=True
    )

    clf = train_model(X_train, y_train)

    train_predictions = inference(clf, X_train)

    precision, recall, fbeta = compute_model_metrics(y_train, train_predictions)

    # All metrics should not be higher then 1
    assert precision <= 1.0
    assert recall <= 1.0
    assert fbeta <= 1.0

    # Only for precision, it should be higher than 0.5
    # the hazard limit for a binary classification
    assert precision >= 0.5


def test_compute_metrics_by_slice(input_df):

    train, test = input_df

    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=config.cat_features, label=config.TARGET, training=True
    )

    clf = train_model(X_train, y_train)

    metrics_by_slice = compute_metrics_by_slice(
        clf=clf,
        encoder=encoder,
        lb=lb,
        df=train,
        target=config.TARGET,
        cat_columns=config.cat_features,
        output_path=None
    )

    # Checking the type and columns of the output of the function compute_metrics_by_slice
    assert isinstance(metrics_by_slice, pd.DataFrame)
    assert metrics_by_slice.shape[1] == 5
    assert all(_ in ["column", "category", "precision", "recall", "f1"] for _ in metrics_by_slice.columns)
    assert all(_ in metrics_by_slice.columns for _ in ["column", "category", "precision", "recall", "f1"])

    # Checking the number of rows
    nb_categories = 0
    for col in config.cat_features:
        nb_categories += train[col].nunique()

    assert len(metrics_by_slice) == nb_categories
