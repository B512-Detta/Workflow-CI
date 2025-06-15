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

# Suppress warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking URI (local)
mlflow.set_tracking_uri("file:./mlruns")

# Set experiment name
experiment_name = "Titanic_Survival_Prediction"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name}")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name}")
    else:
        experiment_id = "0"
        print("Using default experiment")

mlflow.set_experiment(experiment_name if experiment_name else None)

# Enable autolog untuk Basic level
mlflow.sklearn.autolog()
print("MLflow sklearn autolog enabled")

def main():
    print("Starting MLflow run...")
    
    try:
        with mlflow.start_run() as run:
            print(f"MLflow run started with ID: {run.info.run_id}")
            
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
                    n_samples = 891  # Ukuran seperti dataset Titanic asli
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
                if col != 'Survived':  # Keep target as int
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
            print("Creating pipeline with StandardScaler and LogisticRegression...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    max_iter=1000, 
                    random_state=42,
                    solver='liblinear'
                ))
            ])
            
            # Train model (autolog will handle all logging)
            print("Training model... (autolog active)")
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            print("Making predictions...")
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics for local display
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
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
            
            print(f"\n✓ MLflow autolog handled ALL logging automatically")
            print(f"✓ Model trained using Scikit-Learn Pipeline")
            print(f"✓ MLflow run completed successfully!")
            print(f"Run ID: {run.info.run_id}")
            
    except Exception as e:
        print(f"Error during MLflow run: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
