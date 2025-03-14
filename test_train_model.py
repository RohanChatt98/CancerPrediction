import os
import pickle
import pytest
from train_model import pipeline, X_train, X_test, y_train, y_test  

@pytest.fixture(scope="module")
def artifacts_path():
    """Fixture to set up artifacts path"""
    path = "artifacts"
    os.makedirs(path, exist_ok=True)
    yield path
    # Clean up after the tests
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

@pytest.fixture
def model_file_path(artifacts_path):
    """Fixture to provide the model file path"""
    return os.path.join(artifacts_path, "cancer_detection_pipeline.pkl")

def test_model_training():
    """Test if the model trains without errors"""
    pipeline.fit(X_train, y_train)
    train_accuracy = pipeline.score(X_train, y_train)
    assert train_accuracy > 0.8, "Training accuracy is too low!"

def test_model_saving(model_file_path):
    """Test if the trained model is saved correctly"""
    with open(model_file_path, "wb") as f:
        pickle.dump(pipeline, f)
    
    assert os.path.exists(model_file_path), "Model file was not saved!"

def test_model_loading(model_file_path):
    """Test if the saved model can be loaded and used"""
    with open(model_file_path, "wb") as f:
        pickle.dump(pipeline, f)

    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    predictions = loaded_model.predict(X_test)
    assert len(predictions) == len(y_test), "Prediction length mismatch!"

def test_model_prediction():
    """Test if the model can predict on new data"""
    sample_input = X_test[:1]  # Use a single sample
    prediction = pipeline.predict(sample_input)
    assert prediction[0] in ['Benign', 'Malignant'], f"Prediction '{prediction[0]}' is out of expected range!"

if __name__ == '__main__':
    pytest.main()