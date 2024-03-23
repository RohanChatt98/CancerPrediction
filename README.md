# Cancer Prediction using Machine Learning

## Overview

This dataset was obtained from Kaggle, it contains $569$ samples of cell growth data frokm different patients. Every cell growth is classified into the following two classes:

1. $B$: Benign
2. $M$: Malignant

There are a total of 31 columns in the dataset, among which there is an `id` column which is removed during the preprocessing steps, 29 columns containing the __features__ and `diagnosis` column is the target variable.

I have used [TPOT](https://github.com/EpistasisLab/tpot) package instead of [Scikit Learn](https://github.com/scikit-learn/scikit-learn) as the former is a low-code ML training module which uses genetic algorithm to optimize the pipeline.

## Setting Up the Environment

You need to have `conda` installed on your machine for creation of the environment.

```bash
$ conda create -f environment.yaml
$ conda activate cancer
```
