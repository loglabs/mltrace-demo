from mltrace import Component
from tests import *


class Cleaning(Component):
    def __init__(self):

        super().__init__(
            name="cleaning",
            owner="plumber",
            description="Cleans raw NYC taxicab data",
            tags=["nyc-taxicab"],
            beforeTests=[],
            afterTests=[],
        )


class Featuregen(Component):
    def __init__(self):

        super().__init__(
            name="featuregen",
            owner="spark-gymnast",
            description="Generates features for high tip prediction problem",
            tags=["nyc-taxicab"],
            beforeTests=[],
            afterTests=[OutliersTest],
        )


class TrainTestSplit(Component):
    def __init__(self):

        super().__init__(
            name="splitting",
            owner="fission",
            description="Splits data into training and test sets",
            tags=["nyc-taxicab"],
            beforeTests=[],
            afterTests=[TrainingAssumptionsTest],
        )


class Training(Component):
    def __init__(self):

        super().__init__(
            name="training",
            owner="personal-trainer",
            description="Trains model for high tip prediction problem",
            tags=["nyc-taxicab"],
            beforeTests=[],
            afterTests=[ModelIntegrityTest],
        )


class Inference(Component):
    def __init__(self):

        super().__init__(
            name="inference",
            owner="sherlock-holmes",
            description="Predicts high tip probability",
            tags=["nyc-taxicab"],
            beforeTests=[],
            afterTests=[],
        )
