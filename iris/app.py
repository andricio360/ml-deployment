"""
This module is the main entry point for the FastAPI application.
"""

from fastapi import FastAPI

from iris import iris_logger
from iris.router import router as iris_router

app = FastAPI()
app.include_router(iris_router, prefix="/iris")


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    """
    Healthcheck endpoint for the Iris classifier.
    """
    try:
        return "Iris classifier for the ML Challenge is ready to go!"
    except Exception as e:
        iris_logger.error(f"An error occurred: {str(e)}")
        return "An error occurred during healthcheck."
