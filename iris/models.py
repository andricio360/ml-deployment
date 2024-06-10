"""
This module contains the IrisClassifier class which is used to classify iris flowers into their species. The class uses a decision tree
classifier, a logistic regression classifier, and a k-nearest neighbors classifier to classify the iris flowers. The class also contains
methods to save the trained models and to classify iris flowers based on the trained models.
"""

import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from iris import iris_logger
from iris.irismlflow import get_mlflow_experiment
from sklearn import datasets, neighbors, tree
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class IrisClassifier:
    """Class to classify iris flowers into their species using
    a decision tree classifier, a logistic regression classifier,
    and a k-nearest neighbors classifier.

    Attributes:
    -----------
    X (array)
        The features of the iris dataset.
    y (array)
        The target values of the iris dataset.
    iris_type (dict)
        A dictionary to map the target values to the iris species.
    model_name (str)
        The name of the model to use for classifying the iris flowers.

    Methods:
    --------
    train_model_logistic : Trains the logistic regression classifier
        using the iris dataset.

    train_model_kneighbors : Trains the k-nearest neighbors classifier
        using the iris dataset.

    train_model_decisiontree : Trains the decision tree classifier
        using the iris dataset.

    save_models : Saves the trained models to disk.

    classify_iris: Classifies the iris flowers into their species
        using the trained models.
    """

    def __init__(self, model_name: str):
        """
        Constructor method to initialize the IrisClassifier class.

        Args:
        -----
        model_name (str) : The name of the model to use for classifying the iris flowers.
        Must be one of 'logistic', 'kneighbors', or 'decisiontree'.
        """
        # self.X, self.y = datasets.load_iris(return_X_y=True)
        self.iris_type = {0: "setosa", 1: "versicolor", 2: "virginica"}
        self.model_name = model_name
        iris = load_iris()
        data = pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
        )
        data.rename(columns={"sepal length (cm)": "sepal_l"}, inplace=True)
        data.rename(columns={"sepal width (cm)": "sepal_w"}, inplace=True)
        data.rename(columns={"petal length (cm)": "petal_l"}, inplace=True)
        data.rename(columns={"petal width (cm)": "petal_w"}, inplace=True)
        self.X = data[["sepal_l", "sepal_w", "petal_l", "petal_w"]]
        self.y = data["target"]

    def train_model_logistic(self) -> LogisticRegression:
        """
        Trains the logistic regression classifier using the iris dataset.

        Returns:
        --------
        LogisticRegression : The trained logistic regression classifier.
        """
        logistic = LogisticRegression(solver="lbfgs", max_iter=1000, multi_class="multinomial").fit(
            self.X, self.y
        )
        self.save_models(logistic)
        return logistic

    def train_model_kneighbors(self) -> neighbors.KNeighborsClassifier:
        """
        Trains the k-nearest neighbors classifier using the iris dataset.

        Returns:
        --------
        KNeighborsClassifier : The trained k-nearest neighbors classifier.
        """
        kneighbors = neighbors.KNeighborsClassifier(n_neighbors=3).fit(self.X, self.y)
        self.save_models(kneighbors)
        return kneighbors

    def train_model_decisiontree(self) -> tree.DecisionTreeClassifier:
        """
        Trains the decision tree classifier using the iris dataset.

        Returns:
        --------
        DecisionTreeClassifier : The trained decision tree classifier.
        """
        decision_classifier = tree.DecisionTreeClassifier().fit(self.X, self.y)
        self.save_models(decision_classifier)
        return decision_classifier

    def save_models(self, model: str) -> None:
        """
        Saves the trained models to disk.

        Args:
        -----
        model (str) : The name of the model to save.
        """
        joblib.dump(model, f"{model}.joblib")

    def detect_drift(self, current_features: dict[str, float]) -> dict:
        """
        Detects drift in the iris dataset.

        Returns:
        --------
        Report : The report containing the drift detection results.
        """
        reference_features_df = self.X
        current_features_df = pd.DataFrame(
            data=current_features, columns=["sepal_l", "sepal_w", "petal_l", "petal_w"], index=[0]
        )
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference_features_df, current_data=current_features_df
        )
        drifted_columns = {}
        for metric in data_drift_report.as_dict().get("metrics"):
            if "drift_by_columns" in metric["result"]:
                for column, details in metric["result"]["drift_by_columns"].items():
                    if details["drift_detected"]:
                        if "drifted_columns_names" not in drifted_columns:
                            drifted_columns["drifted_columns_names"] = []
                        drifted_columns["drifted_columns_names"].append(column)

        return_dict = data_drift_report.as_dict().get("metrics")[0].get("result")
        if drifted_columns:
            return_dict.update(drifted_columns)
        return return_dict

    def classify_iris(self, features: dict) -> dict:
        """
        Classifies the iris flowers into their species using the trained models.

        Args:
        -----
        features (dict) : A dictionary containing the features of the iris flowers.

        Returns:
        --------
        dict : A dictionary containing the classification results.

        Raises:
        -------
        Exception : If an error occurs during the classification process.
        """
        try:

            get_mlflow_experiment()
            iris_logger.info(f"Classifying iris with model: {self.model_name}")
            iris_logger.info(f"Features: {features}")
            method = joblib.load(f"iris/models/{self.model_name}.joblib")
            X = [
                features["sepal_l"],
                features["sepal_w"],
                features["petal_l"],
                features["petal_w"],
            ]
            drift_dict = self.detect_drift(features)
            prediction = method.predict_proba([X])
            prediction_output = method.predict([X])
            class_prob = self.iris_type[np.argmax(prediction)]
            probability = round(max(prediction[0]), 2)
            output = int(prediction_output[0])
            mlflow.log_params(
                {
                    "class": class_prob,
                    "probability": probability,
                    "output": output,
                }
            )
            mlflow.set_tags(
                {
                    "model_name": self.model_name,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                }
            )
            iris_logger.info("Logged parameters to mlflow")
            return_dict = {
                "class": class_prob,
                "probability": probability,
                "output": output,
            }
            iris_logger.info(f"Drift Report: {drift_dict}")
            mlflow.log_dict(drift_dict, "drift_report")
            return return_dict
        except Exception as e:
            iris_logger.error(f"Error: {e}")
            return {"error": f"Error: {e}"}
