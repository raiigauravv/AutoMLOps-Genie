# 📄 pipelines/pipeline_builder.py
# AutoML pipeline: AutoGluon, MLflow logging, SHAP explainability

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import joblib
from datetime import datetime
from autogluon.tabular import TabularPredictor
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

def run_autogluon_isolated(df, target_col, problem_type, save_path):
    """
    Run AutoGluon in a subprocess to completely isolate state.
    This is a fallback method to avoid 'Learner is already fit' errors.
    """
    import subprocess
    import json
    import tempfile
    
    try:
        # Save dataframe to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            df.to_csv(temp_csv, index=False)
        
        # Run AutoGluon in subprocess
        cmd = [
            sys.executable, 
            "isolated_autogluon.py",
            temp_csv,
            target_col, 
            problem_type,
            save_path
        ]
        
        print(f"Running isolated AutoGluon: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Clean up temp file
        os.unlink(temp_csv)
        
        if result.returncode != 0:
            print(f"Subprocess error: {result.stderr}")
            raise Exception(f"AutoGluon subprocess failed: {result.stderr}")
        
        # Parse results
        results = json.loads(result.stdout)
        
        if not results["success"]:
            raise Exception(f"AutoGluon failed: {results.get('error', 'Unknown error')}")
        
        # Convert leaderboard back to DataFrame
        leaderboard_data = results["leaderboard"]
        leaderboard = pd.DataFrame(leaderboard_data)
        
        return results["best_model"], results["score"], leaderboard
        
    except Exception as e:
        print(f"Isolated AutoGluon failed: {e}")
        raise e
    """Reset AutoGluon environment to prevent state conflicts."""
    try:
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any matplotlib state
        plt.close('all')
        
        # Clear any AutoGluon caches if they exist
        # This is a precautionary measure
        import tempfile
        import shutil
        temp_dir = tempfile.gettempdir()
        
        # Look for any AutoGluon temp files and clean them
        for item in os.listdir(temp_dir):
            if 'autogluon' in item.lower() or 'tabular' in item.lower():
                try:
                    item_path = os.path.join(temp_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except:
                    pass  # Ignore cleanup errors
                    
        print("AutoGluon environment reset completed")
        
    except Exception as e:
        print(f"Environment reset warning: {e}")
        # Don't fail if reset has issues

def run_pipeline(df: pd.DataFrame, target_col: str, task_type: str = None):
    """
    Run AutoML pipeline with AutoGluon, log to MLflow, save best model.
    Returns: (result string, model_path, leaderboard DataFrame)
    """
    print(f"Starting pipeline with target: {target_col}, task_type: {task_type}")
    print(f"Dataset shape: {df.shape}")
    
    # Clean up any existing MLflow runs to avoid state conflicts
    try:
        mlflow.end_run()  # End any existing run
    except:
        pass
    
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("AutoMLOps-Genie")

    # Validate input data
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    if df[target_col].isna().sum() > 0:
        print(f"Warning: Target column has {df[target_col].isna().sum()} missing values")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Task type detection for AutoGluon
    if not task_type:
        task_type = "classification" if y.nunique() <= 10 else "regression"

    # Map task type to AutoGluon problem types
    if task_type == "regression":
        problem_type = "regression"
    else:  # classification
        # Determine if binary or multiclass
        num_classes = y.nunique()
        if num_classes == 2:
            problem_type = "binary"
        else:
            problem_type = "multiclass"
    
    # Create unique save path with microseconds to avoid conflicts
    import uuid
    unique_id = str(uuid.uuid4())[:8]  # Short unique ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"models/ag_{timestamp}_{unique_id}"
    
    # Ensure completely clean directory
    if os.path.exists(save_path):
        import shutil
        print(f"Removing existing directory: {save_path}")
        shutil.rmtree(save_path)
    
    os.makedirs(save_path, exist_ok=True)
    print(f"Created fresh model directory: {save_path}")

    # Run AutoML with progressive fallback approach
    with mlflow.start_run():
        predictor = None
        successful_path = None
        
        # First attempt with original approach
        try:
            print(f"Attempting training with original approach...")
            
            # Force cleanup
            import gc
            gc.collect()
            
            # Create predictor
            predictor = TabularPredictor(
                label=target_col, 
                problem_type=problem_type, 
                path=save_path,
                eval_metric="accuracy" if problem_type in ["binary", "multiclass"] else "root_mean_squared_error",
                verbosity=1
            )
            
            print(f"Starting model training...")
            df_clean = df.copy()
            predictor.fit(
                df_clean, 
                presets="medium_quality_faster_train",
                time_limit=60,
                num_bag_folds=0,  # Disable bagging to prevent complexity
                num_stack_levels=0  # Disable stacking
            )
            
            print(f"Model training completed successfully")
            successful_path = save_path
            
        except Exception as fit_error:
            print(f"Training failed: {fit_error}")
            
            # Clean up failed attempt
            if predictor is not None:
                try:
                    del predictor
                except:
                    pass
            
            if os.path.exists(save_path):
                import shutil
                shutil.rmtree(save_path)
            
            # Re-create directory for next attempt
            os.makedirs(save_path, exist_ok=True)
            
            # Second attempt with minimal settings
            try:
                print(f"Attempting training with minimal settings...")
                import gc
                gc.collect()
                
                predictor = TabularPredictor(
                    label=target_col, 
                    problem_type=problem_type, 
                    path=save_path,
                    verbosity=0  # Silent mode
                )
                
                df_minimal = df.copy()
                predictor.fit(
                    df_minimal, 
                    presets="good_quality_faster_inference",  # Fastest preset
                    time_limit=30,  # Shorter time
                    hyperparameters={'GBM': {}, 'CAT': {}, 'RF': {}}  # Only basic models
                )
                
                print(f"Minimal training completed successfully")
                successful_path = save_path
                
            except Exception as second_error:
                print(f"All training attempts failed. Last error: {second_error}")
                
                # Final cleanup
                if os.path.exists(save_path):
                    import shutil
                    shutil.rmtree(save_path)
                
                raise Exception(f"AutoML training failed after multiple attempts. Last error: {second_error}")

        leaderboard = predictor.leaderboard(silent=True)
        best_model = leaderboard.iloc[0]["model"]

        # Evaluate
        perf = predictor.evaluate(df)
        score = perf["accuracy"] if "accuracy" in perf else perf.get("root_mean_squared_error", None)
        if problem_type in ["binary", "multiclass"]:
            mlflow.log_metric("accuracy", score)
        else:
            mlflow.log_metric("rmse", score)

        mlflow.log_param("target", target_col)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("problem_type", problem_type)
        mlflow.log_param("best_model", best_model)

        # AutoGluon saves the model automatically in the save_path
        # The predictor object itself contains the model path
        model_path = save_path  # This is the directory where AutoGluon saved the model
        
        # Create a simple model info file for download
        model_info_path = os.path.join(save_path, "model_info.txt")
        with open(model_info_path, 'w') as f:
            f.write(f"AutoGluon Model\n")
            f.write(f"Target: {target_col}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Problem Type: {problem_type}\n")
            f.write(f"Best Model: {best_model}\n")
            f.write(f"Score: {score:.4f}\n")
            f.write(f"Model Path: {save_path}\n")
            f.write(f"\nTo load this model:\n")
            f.write(f"from autogluon.tabular import TabularPredictor\n")
            f.write(f"predictor = TabularPredictor.load('{save_path}')\n")
        
        # Log the model directory as an artifact
        mlflow.log_artifacts(save_path)

    return (
        f"Pipeline complete ✅ ({task_type}, best: {best_model}, score: {score:.4f})",
        model_info_path,  # Return the info file for download
        leaderboard[["model", "score_val"]],
        save_path,  # Return the actual model directory path for SHAP
    )

def get_shap_plot(model_path, df, parsed):
    """
    Generate SHAP feature importance plot for the trained AutoGluon model.
    Falls back to AutoGluon's built-in feature importance if SHAP fails.
    Returns a matplotlib figure.
    """
    if not model_path or not os.path.isdir(model_path):
        print(f"SHAP: Model directory doesn't exist: {model_path}")
        return None

    try:
        print(f"SHAP: Loading model from {model_path}")
        # Load the AutoGluon predictor from the saved path
        predictor = TabularPredictor.load(model_path)
        print(f"SHAP: Model loaded successfully")
        
        target = parsed.get("target")
        print(f"SHAP: Target column: {target}")
        X = df.drop(columns=[target])
        print(f"SHAP: Feature matrix shape: {X.shape}")
        
        # Sample data for SHAP (performance optimization)
        sample_size = min(50, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        # Try SHAP first
        try:
            print("Attempting SHAP explanation...")
            
            # Create prediction function wrapper
            def predict_fn(X_input):
                """Wrapper function for SHAP that handles AutoGluon predictions"""
                try:
                    # Convert numpy array to DataFrame if needed
                    if not isinstance(X_input, pd.DataFrame):
                        X_df = pd.DataFrame(X_input, columns=X.columns)
                    else:
                        X_df = X_input
                    
                    # Get predictions - use predict_proba for classification, predict for regression
                    try:
                        # Try predict_proba first (classification)
                        probs = predictor.predict_proba(X_df)
                        # For binary classification, return positive class probabilities
                        if hasattr(probs, 'shape') and len(probs.shape) == 2 and probs.shape[1] == 2:
                            return probs.iloc[:, 1].values if hasattr(probs, 'iloc') else probs[:, 1]
                        return probs.values if hasattr(probs, 'values') else probs
                    except:
                        # Fallback to regular predict (regression or if predict_proba fails)
                        preds = predictor.predict(X_df)
                        return preds.values if hasattr(preds, 'values') else preds
                        
                except Exception as pred_err:
                    print(f"Prediction wrapper error: {pred_err}")
                    raise pred_err
            
            # Use KernelExplainer for robustness across different model types
            background_sample = X_sample.iloc[:min(20, len(X_sample))]  # Small background sample
            explainer = shap.KernelExplainer(predict_fn, background_sample)
            
            # Calculate SHAP values for a subset
            explain_sample = X_sample.iloc[:min(10, len(X_sample))]
            shap_values = explainer.shap_values(explain_sample)
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 6))
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - use first class or positive class
                plot_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                plot_values = shap_values
            
            # Create bar plot of mean absolute SHAP values
            mean_shap = pd.Series(
                np.abs(plot_values).mean(axis=0),
                index=explain_sample.columns
            ).sort_values(ascending=True)
            
            # Plot top 10 features
            top_features = mean_shap.tail(10)
            plt.barh(range(len(top_features)), top_features.values, color='skyblue', alpha=0.8)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Mean |SHAP Value|')
            plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            print("SHAP plot generated successfully")
            fig = plt.gcf()
            plt.close()
            return fig
            
        except Exception as shap_error:
            print(f"SHAP failed: {shap_error}")
            print("Falling back to AutoGluon feature importance...")
            
            # Fallback to AutoGluon's built-in feature importance
            try:
                importance_df = predictor.feature_importance(X_sample)
                
                if importance_df is not None and not importance_df.empty:
                    plt.figure(figsize=(10, 6))
                    
                    # Get top 10 features
                    top_importance = importance_df.head(10).sort_values('importance', ascending=True)
                    
                    plt.barh(range(len(top_importance)), top_importance['importance'], 
                            color='lightcoral', alpha=0.8)
                    plt.yticks(range(len(top_importance)), top_importance['feature'])
                    plt.xlabel('Feature Importance')
                    plt.title('Model Feature Importance', fontsize=14, fontweight='bold')
                    plt.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    
                    print("AutoGluon feature importance plot generated")
                    fig = plt.gcf()
                    plt.close()
                    return fig
                else:
                    print("AutoGluon feature importance returned empty results")
                    
            except Exception as ag_error:
                print(f"AutoGluon feature importance failed: {ag_error}")
            
            # Final fallback: correlation-based importance
            print("Using correlation-based fallback...")
            return create_correlation_plot(df, target)
            
    except Exception as e:
        print(f"Overall get_shap_plot error: {e}")
        print(f"Model path: {model_path}")
        print(f"Model path exists: {os.path.isdir(model_path) if model_path else 'None'}")
        if model_path and os.path.isdir(model_path):
            print(f"Contents of model directory: {os.listdir(model_path)}")
        return None

def create_correlation_plot(df, target):
    """
    Create a correlation-based feature importance plot as final fallback.
    """
    try:
        # Handle categorical variables by encoding them
        df_encoded = df.copy()
        
        # Encode categorical variables
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != target:  # Don't encode the target yet
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Encode target if it's categorical
        if df_encoded[target].dtype == 'object':
            le_target = LabelEncoder()
            df_encoded[target] = le_target.fit_transform(df_encoded[target].astype(str))
        
        # Calculate correlations
        correlations = df_encoded.corr()[target].abs().sort_values(ascending=False)
        # Remove target self-correlation and get top features
        feature_correlations = correlations.drop(target).head(10).sort_values(ascending=True)
        
        if len(feature_correlations) > 0:
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_correlations)), feature_correlations.values,
                    color='gold', alpha=0.8)
            plt.yticks(range(len(feature_correlations)), feature_correlations.index)
            plt.xlabel('Absolute Correlation with Target')
            plt.title('Feature Importance (Correlation-based)', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            print("Correlation-based feature importance plot generated")
            fig = plt.gcf()
            plt.close()
            return fig
        else:
            print("No valid correlations found")
            return None
        
    except Exception as e:
        print(f"Correlation fallback error: {e}")
        return None

def load_recent_mlflow_runs(limit: int = 5):
    """Load recent MLflow experiment runs with flexible metric handling."""
    try:
        exp = mlflow.get_experiment_by_name("AutoMLOps-Genie")
        if exp:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if runs.empty:
                return pd.DataFrame()
            
            # Base columns that should always exist
            base_cols = ["run_id", "start_time"]
            
            # Parameter columns (optional)
            param_cols = []
            for col in ["params.target", "params.task_type", "params.problem_type"]:
                if col in runs.columns:
                    param_cols.append(col)
            
            # Metric columns (optional)
            metric_cols = []
            for col in ["metrics.accuracy", "metrics.rmse"]:
                if col in runs.columns:
                    metric_cols.append(col)
            
            # Combine all available columns
            available_cols = base_cols + param_cols + metric_cols
            
            return runs[available_cols].sort_values("start_time", ascending=False).head(limit)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading MLflow runs: {e}")
        return pd.DataFrame()
