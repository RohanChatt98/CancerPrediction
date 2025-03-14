import pandas as pd
import json
import joblib
import os
from sklearn.pipeline import Pipeline

# Environment variables from CodeBuild or other environment settings
MODEL_PATH = os.getenv('MODEL_PATH', 'cancer_detection_pipeline.pkl')  # Path to your saved model, set by CodeBuild environment

def load_model(model_path):
    """
    Function to load the trained model pipeline.
    """
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def prepare_input_data():
    """
    Function to prepare the sample input data.
    """
    input_data_json = '''{
      "radius_mean": 14.0,
      "texture_mean": 19.0,
      "perimeter_mean": 90.0,
      "area_mean": 500.0,
      "smoothness_mean": 0.08,
      "compactness_mean": 0.15,
      "concavity_mean": 0.13,
      "points_mean": 0.03,
      "symmetry_mean": 0.18,
      "dimension_mean": 0.06,
      "radius_se": 0.4,
      "texture_se": 1.2,
      "perimeter_se": 2.5,
      "area_se": 30.0,
      "smoothness_se": 0.02,
      "compactness_se": 0.04,
      "concavity_se": 0.05,
      "points_se": 0.01,
      "symmetry_se": 0.03,
      "dimension_se": 0.02,
      "radius_worst": 18.0,
      "texture_worst": 25.0,
      "perimeter_worst": 120.0,
      "area_worst": 800.0,
      "smoothness_worst": 0.15,
      "compactness_worst": 0.30,
      "concavity_worst": 0.28,
      "points_worst": 0.05,
      "symmetry_worst": 0.22,
      "dimension_worst": 0.09,
      "Sex": "F"
    }'''

    # Convert JSON to a Python dictionary
    input_data = json.loads(input_data_json)

    # Convert dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    return input_df

def run_prediction(model, input_data):
    """
    Function to run prediction on the input data.
    """
    try:
        prediction = model.predict(input_data)
        print("Prediction result:", prediction)
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def test_model():
    """
    The main test function to ensure the model is working properly.
    """
    try:
        # Load the trained model
        model = load_model(MODEL_PATH)

        # Prepare sample input data
        input_data = prepare_input_data()

        # Run prediction
        prediction = run_prediction(model, input_data)

        # Validate the output (basic validation)
        if prediction is not None:
            print("Model prediction completed successfully.")
        else:
            print("Model prediction failed.")
            raise ValueError("Prediction returned None.")
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == '__main__':
    test_model()
