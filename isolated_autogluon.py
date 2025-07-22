#!/usr/bin/env python3
"""
Isolated AutoGluon runner to avoid state conflicts.
This script runs in a separate process to ensure clean state.
"""
import sys
import json
import pandas as pd
from autogluon.tabular import TabularPredictor
import os

def run_isolated_autogluon(data_file, target_col, problem_type, output_dir):
    """Run AutoGluon in isolation."""
    try:
        # Load data
        df = pd.read_csv(data_file)
        
        # Create predictor
        predictor = TabularPredictor(
            label=target_col,
            problem_type=problem_type,
            path=output_dir,
            eval_metric="accuracy" if problem_type in ["binary", "multiclass"] else "root_mean_squared_error"
        )
        
        # Train model
        predictor.fit(
            df,
            presets="medium_quality_faster_train",
            time_limit=60,
            num_bag_folds=0,
            num_stack_levels=0
        )
        
        # Get leaderboard
        leaderboard = predictor.leaderboard(silent=True)
        best_model = leaderboard.iloc[0]["model"]
        
        # Evaluate
        perf = predictor.evaluate(df)
        if problem_type in ["binary", "multiclass"]:
            score = perf.get("accuracy", 0.0)
        else:
            score = perf.get("root_mean_squared_error", 0.0)
        
        # Return results
        results = {
            "success": True,
            "best_model": best_model,
            "score": float(score),
            "leaderboard": leaderboard[["model", "score_val"]].to_dict('records')
        }
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": str(type(e))
        }

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(json.dumps({"success": False, "error": "Invalid arguments"}))
        sys.exit(1)
    
    data_file, target_col, problem_type, output_dir = sys.argv[1:5]
    results = run_isolated_autogluon(data_file, target_col, problem_type, output_dir)
    print(json.dumps(results, default=str))
