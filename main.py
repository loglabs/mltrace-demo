import argparse
import pandas as pd
import typing

from components import *
from datetime import datetime
from utils import io, feature_generators, models

parser = argparse.ArgumentParser(description="Run inference.")
parser.add_argument("--start", type=str, help="Start date", nargs="?")
parser.add_argument("--end", type=str, help="End date", nargs="?")
parser.add_argument(
    "--mode",
    type=str,
    help="training or inference",
    const="inference",
    nargs="?",
)
args = parser.parse_args()


def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Format: MM/DD/YYYY
    """
    # Iterate through months and years between start and end dates
    start_date = datetime.strptime(start_date, "%m/%d/%Y")
    end_date = datetime.strptime(end_date, "%m/%d/%Y")

    assert end_date >= start_date
    assert end_date.year == 2020
    assert start_date.month >= 1
    assert start_date.month <= 5

    dfs = []

    for month in range(start_date.month, start_date.month + 1):
        df = pd.read_parquet("data/jan.pq")
        if month == 2:
            df = pd.read_parquet("data/feb.pq")
        elif month == 3:
            df = pd.read_parquet("data/march.pq")
        elif month == 4:
            df = pd.read_parquet("data/april.pq")
        elif month == 5:
            df = pd.read_parquet("data/may.pq")

    df = pd.concat(dfs)
    return df


def clean_data(
    df: pd.DataFrame, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    This function removes rows with negligible fare amounts and out of bounds of the start and end dates.

    Args:
        df: pd dataframe representing data
        start_date (optional): minimum date in the resulting dataframe
        end_date (optional): maximum date in the resulting dataframe (not inclusive)

    Returns:
        pd: DataFrame representing the cleaned dataframe
    """
    df = df[df.fare_amount > 5]  # avoid divide-by-zero
    if start_date:
        df = df[df.tpep_dropoff_datetime.dt.strftime("%m/%d/%Y") >= start_date]
    if end_date:
        df = df[df.tpep_dropoff_datetime.dt.strftime("%m/%d/%Y") < end_date]

    clean_df = df.reset_index(drop=True)
    return clean_df


@Featuregen().run(auto_log=True)
def featurize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function constructs features from the dataframe.
    """
    pickup_features = feature_generators.Pickup().compute(df)

    trip_features = feature_generators.Trip().compute(df)
    categorical_features = feature_generators.Categorical().compute(df)
    label = feature_generators.HighTip().compute(df, tip_fraction=0.1)

    # Concatenate features
    features_df = pd.concat(
        [
            pickup_features,
            trip_features,
            categorical_features,
            label,
            df["tpep_pickup_datetime"].to_frame(),
        ],
        axis=1,
    )

    return features_df


@TrainTestSplit().run(auto_log=True)
def train_test_split(
    df: pd.DataFrame,
) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function splits the dataframe into train and test.
    """
    # Split into train and test
    date_column = "tpep_pickup_datetime"
    label_column = "high_tip_indicator"
    df = df.sort_values(by=date_column, ascending=True)
    train_df, test_df = (
        df.iloc[: int(len(df) * 0.8)],
        df.iloc[int(len(df) * 0.8) :],
    )

    return train_df, test_df


@Training().run(auto_log=True, output_vars=["rfc_path"])
def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "high_tip_indicator",
) -> None:
    """
    This function runs training on the dataframe.
    """

    feature_columns = [
        "pickup_weekday",
        "pickup_hour",
        "pickup_minute",
        "work_hours",
        "passenger_count",
        "trip_distance",
        # "PULocationID",
        # "DOLocationID",
        "RatecodeID",
        "congestion_surcharge",
        "loc_code_diffs",
        # "trip_speed"
    ]

    params = {"max_depth": 4, "n_estimators": 10}

    # Create and train model
    mw = models.RandomForestModelWrapper(
        feature_columns=feature_columns, model_params=params
    )
    mw.add_data_path("train_df", "train_df")
    mw.add_data_path("test_df", "test_df")
    mw.train(train_df, label_column)

    # Score model
    train_scores = mw.score(train_df, label_column)
    test_scores = mw.score(test_df, label_column)
    mw.add_metrics(train_scores)
    mw.add_metrics(test_scores)

    # Print paths and metrics
    print("Metrics:")
    print(mw.get_metrics())

    # Print feature importances
    feature_importances = mw.get_feature_importances()
    print(feature_importances)

    # Save model
    rfc_path = mw.save("training/models/tip")
    print(f"Saved {rfc_path}")


@Inference().run(auto_log=True, input_vars=["rfc_path"])
def inference(features_df: pd.DataFrame):
    """
    This function runs inference on the dataframe.
    """
    # Load model
    mw = models.RandomForestModelWrapper.load("training/models/tip")
    rfc_path = io.get_output_path("training/models/tip")

    # Predict
    predictions = mw.predict(features_df)
    scores = mw.score(features_df, "high_tip_indicator")
    predictions_df = features_df
    predictions_df["prediction"] = predictions

    return predictions_df, scores


##################### PIPELINE CODE #############################

if __name__ == "__main__":
    mode = args.mode if args.mode else "inference"
    start_date = args.start if args.start else "01/01/2020"
    end_date = args.end if args.end else "01/31/2020"

    # Clean and featurize data
    df = load_data(start_date, end_date)
    clean_df = clean_data(df, start_date, end_date)
    features_df = featurize_data(clean_df)

    # If training, train a model and save it
    if mode == "training":
        train_df, test_df = train_test_split(features_df)
        train_model(train_df, test_df)

    # If inference, load the model and make predictions
    elif mode == "inference":
        predictions, scores = inference(features_df)
        print(scores)

    else:
        print(f"Mode {mode} not supported.")
