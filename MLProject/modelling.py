import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import warnings
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Enable autolog (ini yang paling penting untuk Basic level)
mlflow.sklearn.autolog()
print("MLflow sklearn autolog enabled")

print("Starting training...")

# Load data
print("Loading data...")
try:
    # Coba berbagai lokasi file
    possible_files = [
        "data/train_clean.csv",
        "train_clean.csv", 
        "../data/train_clean.csv"
    ]
    
    df = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Data loaded from: {file_path}")
            break
    
    if df is None:
        # Create sample data for CI testing
        print("No data file found. Creating sample data for CI testing...")
        np.random.seed(42)
        n_samples = 891
        df = pd.DataFrame({
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Sex': np.random.choice([0, 1], n_samples),
            'Age': np.random.normal(30, 10, n_samples),
            'Fare': np.random.exponential(30, n_samples),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.4, n_samples),
            'Survived': np.random.choice([0, 1], n_samples)
        })
        print("Sample data created for CI testing")
        
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Basic data info  
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert integer columns to float for consistency
for col in df.select_dtypes(include=['int']).columns:
    if col != 'Survived':
        df[col] = df[col].astype('float64')

# Separate features and target
target_col = 'Survived'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Pipeline
print("Creating pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=1000, 
        random_state=42,
        solver='liblinear'
    ))
])

# Train model (autolog akan menangani semua logging otomatis)
print("Training model... (autolog akan menangani logging)")
pipeline.fit(X_train, y_train)

# Make predictions untuk display
y_pred = pipeline.predict(X_test)

# Calculate metrics untuk display
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Print results
print(f"\n=== Model Performance ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print(f"\n Training completed successfully!")
print(f"MLflow autolog handled all logging automatically")
