import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

cancerdata = pd.read_csv("cancerdata.csv")

# Data Preprocessing
cancerdata['diagnosis'] = cancerdata['diagnosis'].replace({'B': 'Benign', 'M': 'Malignant'})
cancerdata.drop(['id'], axis=1, inplace=True)

# Define features and target
X = cancerdata.drop(columns=['diagnosis'])
y = cancerdata['diagnosis']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(exclude=['object']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define the transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))  # One-hot encode
        ]), categorical_features),
        
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute missing numerical values
            ('scaler', MinMaxScaler())  # scale numerical features
        ]), numeric_features)
    ]
)
# Create the pipeline with preprocessing and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())  # Replace with your trained model
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, "cancer_detection_pipeline.pkl")
print("Pipeline trained and saved as cancer_detection_pipeline")