from re import I
import sys
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", help="csv file cotaining names")
    parser.add_argument("-r", "--ratio", help="define split ratio between training and validation", default=0.1, type=float)
    args = parser.parse_args()

    names_df = pd.read_csv(args.names)

    validation_df = names_df.sample(frac=args.ratio)
    training_df = names_df.drop(validation_df.index)

    print(f"Writing {training_df.shape[0]} names to training.csv")
    training_df.to_csv('training.csv', index=False)
    print(f"Writing {validation_df.shape[0]} names to validation.csv")
    validation_df.to_csv('validation.csv', index=False)