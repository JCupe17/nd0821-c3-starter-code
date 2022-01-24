TEST_SIZE = 0.2
TARGET = "salary"
MODEL_PATH = "model/classifier.pkl"
DATA_PATH = "data/census_cleaned.csv"
METRICS_PATH = "model/metrics_by_slice.csv"

all_columns = [
    "age", "workclass", "fnlgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]

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