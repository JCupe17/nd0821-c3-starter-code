import pytest

from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app


# Instantiate the testing client with our app.
@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


# Write tests using the same syntax as with the requests' module.
def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': 'Welcome to the salary predictor!!'}


def test_wrong_get(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post(client):

    r = client.post("/predictions", json={
        "age": 39,
        "fnlgt": 77516,
        "workclass": "State-gov",
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"predicted salary": "<=50K"}


def test_wrong_post(client):
    r = client.post("/predictions", json={
        "age": 40,
        "fnlgt": 154374,
        "workclass": "Unknown",
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Machine-op-inspct",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 422
