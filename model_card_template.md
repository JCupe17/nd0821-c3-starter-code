# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The code in this repository presents a training pipeline for a **Binary classification**.
It uses the Random Forest classifier from the scikit-learn library. It was fitted to learn
the classification of a person predicting if his or her income is higher or lower than 50K
per year.

The model was trained with the hyper-parameters by default. A grid search could be done to
improve the performance of the model.

## Intended Use

The goal of the model is to predict the salary of a person based on some social-economics
characteristics.

## Training Data

The training data `census.csv` is a well-know dataset that is described using pandas
profiling [here](
https://pandas-profiling.github.io/pandas-profiling/examples/master/census/census_report.html).

This dataset has **14 variables**: 6 numerical and 8 categorical. The total number of
observations are 32561.

### Data cleaning

The main difference between the given dataset in the project is a little different from the
raw dataset presented by the report of pandas profiling. Indeed, the `census.csv` file
contains white spaces which were removed. The new file `census_cleaned.csv` contains a cleaned
data without any white space, and this file is used to train the Random Forest model.

### Data splitting

To evaluate the trained model, the dataset has been split in two datasets: train and test sets,
with a 80/20 ratio between both datasets. For this split, we have used the train_test_split
function from scikit-learn.

### Training pipeline

To train the model using a Random Forest model from scikit-learn, we should pre-process the
different variables specially the categorical variables. Indeed, the values of categorical values
are strings, and they cannot be used directly in the model. For that, we perform two different
operations:

* We apply a **label encoder** on the target variable.
* We apply a **one-hot encoder** to all categorical variables.

The encoder considers only the values found in the training dataset. If there is a new value in
the test dataset, this new value is ignored.

## Evaluation Data

The evaluation of the data is done in the test dataset.

## Metrics

Three different metrics have been used in this project:

* **Precision:** is the fraction of relevant instances among the retrieved instances.
* **Recall:** is the fraction of relevant instances that were retrieved.
* **fbeta (beta=1) or F1-score:** is a measure of the accuracy of the test. If beta is equal to 1,
    it is the F1-score which is the harmonic mean of the precision and recall.

The metrics for the predicted model are: `Precision: 0.73`, `Recall: 0.62`, and `1-score: 0.67`.

To better analyze the results of the model, another calculation by category has been also
implemented (see `model/metrics_by_slice.csv`). Using this file, we can better understand the
behavior of the model for a given value of any categorical variable.

## Ethical Considerations

Given that the raw dataset contains census information from the US only, the trained model
could be only applied for american people or residents in the US. Furthermore, this model
could have some biases based on sex, race, native-country, and age. Indeed, the dataset has
unbalanced categories for race, native-country, and sex variables.

The predictions of the model should be taken carefully according the people on whom this
model is used.

## Caveats and Recommendations

The model is limited to the US residents. Furthermore, the raw dataset is unbalanced for
some variables. Some solutions to avoid any bias in the trained model is to balance the
dataset or to train the model using up sampling and down sampling techniques such as SMOTE.
