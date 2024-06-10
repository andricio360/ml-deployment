import mlflow
from fastapi import APIRouter
from iris import iris_logger
from iris.irismlflow import get_mlflow_experiment
from iris.models import IrisClassifier
from starlette.responses import JSONResponse

router = APIRouter()


@router.post("/classify_iris/{model_name}")
def classify_iris(iris_features: dict, model_name: str) -> JSONResponse:
    """
    Classify iris flowers into their species using the specified model.

    Args:
    -----
    iris_features (dict) : The features of the iris flower.
    model_name (str) : The name of the model to use for classifying the iris flowers.
    Must be one of 'logistic', 'kneighbors', or 'decisiontree'.

    Returns:
    --------
    JSONResponse : The classification of the iris flower.
    """
    get_mlflow_experiment()
    with mlflow.start_run():
        iris_classifier = IrisClassifier(model_name)
        prediction = iris_classifier.classify_iris(iris_features)
        iris_logger.info(f"Classifying iris with model: {model_name}")
        iris_logger.info(f"Features: {iris_features}")
        iris_logger.info(f"Prediction: {prediction}")
        mlflow.log_params(
            {
                "sepal_l": iris_features.get("sepal_l"),
                "sepal_w": iris_features.get("sepal_w"),
                "petal_l": iris_features.get("petal_l"),
                "petal_w": iris_features.get("petal_w"),
            }
        )
        # mlflow.set_tag("run_id", irismlflow.run_id)
    return JSONResponse(prediction)
