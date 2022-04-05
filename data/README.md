# Training Data
The training data used in this repository is derived from [here](https://github.com/fivethirtyeight/data/tree/master/most-common-name
) making use of the top 100 most common first names and surnames.

Based on the original `firstnames.csv` and `lastnames.csv` a combination `names.csv`
was created that combines each surname with a random selection of 10 first names,
resulting in 1000 name combinations.

This training data is than split 9/1 into `training.csv` and `validation.csv`.
