# Put the code for your API here.
import numpy as np
import pandas as pd

import pickle as pkl
from fastapi import FastAPI
from web_app import InputData

import starter.config as config
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI(
    title="API for salary predictor",
    description="This API helps to classify",
    version="0.0.1",
)


@app.get("/")
async def welcome():
    return {'message': 'Welcome to the salary predictor!!'}


@app.post("/predictions")
async def prediction(input_data: InputData):

    # Read the trained model and the encoder
    with open(config.MODEL_PATH, 'rb') as f:
        encoder, lb, model = pkl.load(f)

    # Formatting input_data
    input_df = pd.DataFrame(
        {k: v for k, v in input_data.dict().items()}, index=[0]
    )
    input_df.columns = [_.replace('_', '-') for _ in input_df.columns]

    # Processing input data
    X, _, _, _ = process_data(
        X=input_df,
        label=None,
        training=False,
        categorical_features=config.cat_features,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]

    return {"predicted salary": y}
