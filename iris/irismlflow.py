"""
Module to initialize and track MLFlow runs for the Iris classification model.
"""

import datetime

import mlflow
from iris import iris_logger


def get_mlflow_experiment():
    """Get the MLFlow experiment to use for
    logging and tracking the Iris classification model.
    """
    experiment_name = "Iris Classification"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        iris_logger.error(f"Experiment '{experiment_name}' not found")
        raise ValueError("Experiment does not exist")


class IrisMlflow:
    """Class to initialize MLFlow experiment and
    track runs for the Iris classification model.

    Attributes:
    ------------
    run_id : str
        The unique identifier for the MLFlow run.

    Methods:
    ---------
    initialize_mlflow : Initializes the MLFlow experiment
        and sets the experiment name and tracking URI.
    """

    def __init__(self):
        """
        Constructor method to initialize the IrisMlflow class.
        """
        self.initialize_mlflow()
        self.run_id = mlflow.active_run().info.run_id


def initialize_mlflow():
    """Initialize the MLFlow experiment to use for
    logging and tracking the Iris classification model.
    """
    mlflow.set_tracking_uri("iris/mlruns")
    mlflow.set_experiment("iris_classification")
    iris_logger.info("MLflow experiment initialized")


def analyze_mlflow():
    """Analyze the MLFlow experiment to track
    runs for the Iris classification model.
    """
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment_id = mlflow.get_experiment_by_name("Iris Classification").experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.date = {datetime.datetime.now().strftime('%Y-%m-%d')}",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    for run in runs:
        run_data = client.get_run(run.info.run_id).data
        params = run_data.params
        # Access run data for analysis
        # params, run_data.metrics, run_data.run_id
        # Compare params against data saved in data model
        # Calculate drift and generate report
        # Save report in logs
        iris_logger.info(f"Drift report for run {run_data.run_id} saved in logs")
