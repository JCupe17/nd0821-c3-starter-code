"""
Schema for the input data using pydantic.

NOTE: pydantic is primarily a parsing library, not a validation library.
Pydantic guarantees the types and constrains of the output model, not the input data.
"""

from typing import Literal
from pydantic import BaseModel


class InputData(BaseModel):
    """
    Schema for the input data on the POST method.
    """
    age: int
    workclass: Literal[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
        "Never-worked"
    ]
    fnlgt: float
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th",
    ]
    education_num: float
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed",
    ]
    occupation: Literal[
        "Adm-clerical",
        "Exec-managerial",
        "Handlers-cleaners",
        "Prof-specialty",
        "Other-service",
        "Sales",
        "Transport-moving",
        "Farming-fishing",
        "Machine-op-inspct",
        "Tech-support",
        "Craft-repair",
        "Protective-serv",
        "Armed-Forces",
        "Priv-house-serv",
    ]
    relationship: Literal[
        "Not-in-family",
        "Husband",
        "Wife",
        "Own-child",
        "Unmarried",
        "Other-relative",
    ]
    race: Literal[
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
    ]
    sex: Literal["Male", "Female"]
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
    ]

    class Config:
        """
        POST method input example
        """

        schema_extra = {
            "example": {
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
        }
