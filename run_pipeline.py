#!/usr/bin/env python3
"""
Quick start script to train and visualize all optimizers.
Runs the complete pipeline: training -> visualization
"""

import subprocess
import sys
import os


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"> {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=cwd or os.path.dirname(__file__) or '.')
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to run: {description}")
        print(f"  Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}")
        return False


def main():
    print("\n")
    print("="*60)
    print("  Neural Network Optimizer Comparison")
    print("  Training & Visualization Pipeline")
    print("="*60)
    
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    results_file = os.path.join(src_dir, 'results.pkl')
    
    if os.path.exists(results_file):
        response = input("\nWARNING: results.pkl already exists. Retrain? (y/n): ").lower()
        if response != 'y':
            print("Skipping training, using existing results...")
        else:
            if not run_command([sys.executable, 'train_and_compare.py'], 
                             "Training models with all optimizers", cwd=src_dir):
                return 1
    else:
        if not run_command([sys.executable, 'train_and_compare.py'], 
                          "Training models with all optimizers", cwd=src_dir):
            return 1
    
    if not run_command([sys.executable, 'visualize_results.py'], 
                      "Generating visualization graphs", cwd=src_dir):
        return 1
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - loss_comparison.png - Training and test loss curves")
    print("  - accuracy_comparison.png - Training and test accuracy")
    print("  - convergence_speed.png - How quickly optimizers converge")
    print("  - final_accuracy_comparison.png - Final performance bar chart")
    print("  - loss_smoothing.png - Smoothed loss trends")
    print("  - results.pkl - Raw training data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
