"""
components.py

This file defines tests to run on inputs and outputs of components.
"""

from mltrace import Test

import pandas as pd
import typing


class OutliersTest(Test):
    def __init__(self):
        super().__init__("Outliers")

    def testComputeStats(self, df: pd.DataFrame):
        """
        Computes stats for all numeric columns in the dataframe.
        """
        # Get numerical columns
        num_df = df.select_dtypes(include=["number"])

        # Compute stats
        stats = num_df.describe()
        print("Dataframe statistics:")
        print(stats)

    def testZScore(
        self,
        df: pd.DataFrame,
        stdev_cutoff: float = 5.0,
        threshold: float = 0.05,
    ):
        """
        Checks to make sure there are no outliers using z score cutoff.
        """
        # Get numerical columns
        num_df = df.select_dtypes(include=["number"])

        z_scores = (
            (num_df - num_df.mean(axis=0, skipna=True))
            / num_df.std(axis=0, skipna=True)
        ).abs()

        if (z_scores > stdev_cutoff).to_numpy().sum() > threshold * len(df):
            print(
                f"Number of outliers: {(z_scores > stdev_cutoff).to_numpy().sum()}"
            )
            print(f"Outlier threshold: {threshold * len(df)}")
            raise Exception("There are outlier values!")


class TrainingAssumptionsTest(Test):
    def __init__(self):
        super().__init__("Training Assumptions")

    # Train-test leakage
    def testLeakage(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, date_column: str
    ):
        """
        Checks to make sure there is no leakage in the training data.
        """
        if train_df[date_column].max() > test_df[date_column].min():
            raise Exception(f"Train and test data are overlapping in dates!")

    # Assess class imbalance
    def testClassImbalance(
        self, train_df: pd.DataFrame, label_column: str, threshold: float = 0.1
    ):
        """
        Checks to make sure there is no class imbalance in the training data.
        """
        frequencies = train_df[label_column].value_counts(normalize=True)
        if frequencies.min() < threshold:
            raise Exception(f"Class imbalance is too high!")


class ModelIntegrityTest(Test):
    def __init__(self):
        super().__init__("Model Integrity")

    def testOverfitting(
        self,
        train_scores: typing.Dict,
        test_scores: typing.Dict,
        threshold: float = 0.5,
    ):
        """
        Test that train and test metrics differences are within a
        threshold of 5 percent.
        """
        for name, val in train_scores.items():
            if abs(val - test_scores[name]) > threshold:
                raise Exception(
                    f"Model overfitted with {name} diff of "
                    + f"{abs(val - test_scores[name])}"
                )

    def testFeatureImportances(
        self,
        feature_importances: pd.DataFrame,
        importance_threshold: float = 0.01,
        num_important_features_threshold: float = 0.5,
    ):
        """Test that feature importances are not heavily skewed."""
        num_unimportant_features = (
            feature_importances["importance"] < importance_threshold
        ).sum()
        if (
            float(num_unimportant_features / len(feature_importances))
            > num_important_features_threshold
        ):
            raise Exception(
                f"{float(num_unimportant_features / len(feature_importances))} "
                + "features are unimportant!"
            )
