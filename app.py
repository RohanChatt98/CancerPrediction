from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the trained pipeline
pipeline = joblib.load("cancer_detection_pipeline.pkl")
# Initialize FastAPI app
app = FastAPI()

# Define request model
class CancerFeatures(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    points_mean: float
    symmetry_mean: float
    dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    points_se: float
    symmetry_se: float
    dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    points_worst: float
    symmetry_worst: float
    dimension_worst: float
    Sex: object

@app.get("/")
def home():
    return {"message": "Welcome to the Cancer Detection API!"}

@app.post("/predict")
def predict(data: CancerFeatures):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Ensure categorical columns are properly handled
    for col in input_data.select_dtypes(include='object').columns:
        input_data[col] = input_data[col].astype(str)

    # Predict using the pipeline
    prediction = pipeline.predict(input_data)

    return {"diagnosis": prediction[0]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
