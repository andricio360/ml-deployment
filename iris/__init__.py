"""
This module is the entry point for the iris package.
"""

import mlflow
from iris.irislogger import IrisLogger
from loguru import logger

iris_logger = IrisLogger(logger)
mlflow.set_tracking_uri("iris/mlruns")
mlflow.set_experiment("Iris Classification")
