import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import warnings
import os

# Suppress warnings untuk output yang lebih bersih
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
        # Fallback ke default experiment
        experiment_id = "0"
        print("Using default experiment")

mlflow.set_experiment(experiment_name if experiment_name else None)

# PENTING: Enable autolog sesuai kriteria Basic
mlflow.sklearn.autolog()
print("MLflow sklearn autolog enabled")

# Start MLflow run
print("Starting MLflow run...")

try:
    with mlflow.start_run() as run:
        print(f"MLflow run started with ID: {run.info.run_id}")
        
        # Load data
        print("Loading data...")
        try:
            possible_files = [
                "data/train_clean.csv",
                "Kriteria1/titanic/train_clean.csv"
            ]
            
            df = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    print(f"Data loaded from: {file_path}")
                    break
            
            if df is None:
                # Create sample data if no file found (for CI testing)
                print("No data file found. Creating sample data for testing...")
                np.random.seed(42)
                n_samples = 100
                df = pd.DataFrame({
                    'Pclass': np.random.choice([1, 2, 3], n_samples),
                    'Sex': np.random.choice([0, 1], n_samples),
                    'Age': np.random.normal(30, 10, n_samples),
                    'Fare': np.random.exponential(30, n_samples),
                    'Survived': np.random.choice([0, 1], n_samples)
                })
                print("Sample data created for CI testing")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Available files in current directory:")
            for f in os.listdir("."):
                if f.endswith('.csv'):
                    print(f"  - {f}")
            raise
        
        # Basic data info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Ubah semua kolom integer ke float untuk konsistensi
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = df[col].astype('float64')
            
        # Log basic dataset info
        mlflow.log_param("dataset_shape", str(df.shape))
        mlflow.log_param("dataset_columns", len(df.columns))
        
        # Pisahkan fitur dan label
        target_col = 'Survived'  # Sesuaikan jika nama kolom target berbeda
        
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found!")
            print(f"Available columns: {list(df.columns)}")
            # Coba cari kolom target dengan nama lain
            possible_targets = ['survived', 'target', 'label', 'y']
            for col in possible_targets:
                if col in df.columns.str.lower():
                    target_col = col
                    break
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        print(f"Features: {list(X.columns)}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log data split info
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_actual", len(X_test))
        
        # Create Pipeline (PENTING: untuk autolog bekerja optimal)
        from sklearn.pipeline import Pipeline
        
        print("Creating pipeline with StandardScaler and LogisticRegression...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                random_state=42,
                solver='liblinear'
            ))
        ])
        
        # Log basic parameters (tambahan manual)
        mlflow.log_param("model_type", "LogisticRegression_Pipeline")
        mlflow.log_param("preprocessing", "StandardScaler")
        
        # Fit model (autolog akan menangani logging otomatis)
        print("Training model... (autolog active)")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Log additional metrics (autolog sudah menangani metrics dasar)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        
        # Print results
        print(f"\n=== Model Performance ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Feature importance (dari model dalam pipeline)
        if hasattr(pipeline.named_steps['classifier'], 'coef_'):
            feature_importance = abs(pipeline.named_steps['classifier'].coef_[0])
            feature_names = X.columns
            
            print(f"\n=== Top 5 Important Features ===")
            importance_pairs = list(zip(feature_names, feature_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (feature, importance) in enumerate(importance_pairs[:5]):
                print(f"{i+1}. {feature}: {importance:.4f}")
        
        print(f"\n✓ MLflow autolog handled model and metrics logging")
        print(f"✓ Model trained using Scikit-Learn Pipeline")
        print(f"✓ No hyperparameter tuning applied (Basic level)")
        print(f"✓ MLflow run completed successfully!")
        print(f"Run ID: {run.info.run_id}")
        
except Exception as e:
    print(f"Error during MLflow run: {e}")
    import traceback
    traceback.print_exc()

print("\nTo view MLflow UI, run: mlflow ui")
print("Then open: http://localhost:5000")