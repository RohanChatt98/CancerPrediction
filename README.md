# Cancer Prediction using Machine Learning

## Overview

This dataset was obtained from Kaggle, it contains $569$ samples of cell growth data frokm different patients. Every cell growth is classified into the following two classes:

1. $B$: Benign
2. $M$: Malignant

There are a total of 31 columns in the dataset, among which there is an `id` column which is removed during the preprocessing steps, 29 columns containing the __features__ and `diagnosis` column is the target variable.

I have used [TPOT](https://github.com/EpistasisLab/tpot) package instead of [Scikit Learn](https://github.com/scikit-learn/scikit-learn) as the former is a low-code ML training module which uses genetic algorithm to optimize the pipeline.

## Setting Up the Environment

create a `conda` environment with python version 3.11 and use `pip install -r requirements.txt` to install the necessary packages. After that start jupyter lab by executing `jupyter lab` command.

## TPOT Settings

the following settings were used:
```json
{"generations": 50,
 "population_size": 50,
 "scoring": "f1_weighted",
 "cv": 5,
 "subsample": 0.5,
 "n_jobs": -1,
 "verbosity": 2,
 "random_state": 1337
}
```

## Results

The `sklearn.metrics.classification_report` on the validation dataset gives the following results
```
              precision    recall  f1-score   support

           0       0.61      0.70      0.65        69
           1       0.40      0.31      0.35        45

    accuracy                           0.54       114
   macro avg       0.50      0.50      0.50       114
weighted avg       0.53      0.54      0.53       114
```