TEST_SIZE = 0.2
TARGET = "salary"
MODEL_PATH = "model/classifier.pkl"
DATA_PATH = "data/census_cleaned.csv"
METRICS_PATH = "model/metrics_by_slice.csv"

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]