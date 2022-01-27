"""This module test the live API deployed in Heroku."""

import json
import requests

app_url = "https://udacity-course3-project.herokuapp.com/"

# Test the GET method
request1 = requests.get(app_url)
assert request1.status_code == 200
print(request1.json())

# Test the POST method
data = {
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
  "native_country": "United-States",
}
request2 = requests.post(f"{app_url}predictions", json=data)

assert request2.status_code == 200
print(request2.json())
