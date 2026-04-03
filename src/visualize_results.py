import matplotlib.pyplot as plt
import pickle
import numpy as np
import os


def load_results(filename='results.pkl'):
    """Load training results from pickle file."""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_loss_comparison(results):
    """Plot training and test loss comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'SGD': '#1f77b4', 'Momentum': '#ff7f0e', 'RMSProp': '#2ca02c', 'Adam': '#d62728'}
    
    for optimizer, color in colors.items():
        axes[0].plot(results[optimizer]['train_losses'], label=optimizer, 
                    linewidth=2.5, color=color, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    for optimizer, color in colors.items():
        axes[1].plot(results[optimizer]['test_losses'], label=optimizer, 
                    linewidth=2.5, color=color, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: output/loss_comparison.png")
    plt.close()


def plot_accuracy_comparison(results):
    """Plot training and test accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'SGD': '#1f77b4', 'Momentum': '#ff7f0e', 'RMSProp': '#2ca02c', 'Adam': '#d62728'}
    
    for optimizer, color in colors.items():
        axes[0].plot(results[optimizer]['train_accuracies'], label=optimizer, 
                    linewidth=2.5, color=color, alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    for optimizer, color in colors.items():
        axes[1].plot(results[optimizer]['test_accuracies'], label=optimizer, 
                    linewidth=2.5, color=color, alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: output/accuracy_comparison.png")
    plt.close()


def plot_convergence_speed(results):
    """Plot convergence speed - how quickly optimizers reach target accuracy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'SGD': '#1f77b4', 'Momentum': '#ff7f0e', 'RMSProp': '#2ca02c', 'Adam': '#d62728'}
    
    x = np.arange(len(results['SGD']['test_accuracies']))
    
    for i, (optimizer, color) in enumerate(colors.items()):
        ax.plot(x, results[optimizer]['test_accuracies'], 
               marker='o', linewidth=2.5, markersize=5, label=optimizer, 
               color=color, alpha=0.8)
    
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% Target')
    ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='90% Target')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.1, 1])
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_speed.png'), dpi=300, bbox_inches='tight')
    print("Saved: output/convergence_speed.png")
    plt.close()


def plot_final_comparison(results):
    """Plot final test accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = list(results.keys())
    final_accuracies = [results[opt]['final_test_accuracy'] for opt in optimizers]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(optimizers, final_accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (bar, acc) in enumerate(zip(bars, final_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.4f}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Final Test Accuracy by Optimizer', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: output/final_accuracy_comparison.png")
    plt.close()


def plot_loss_smoothing(results):
    """Plot smoothed loss for better visibility of trends."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'SGD': '#1f77b4', 'Momentum': '#ff7f0e', 'RMSProp': '#2ca02c', 'Adam': '#d62728'}
    
    def smooth(y, box_pts=3):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')
    
    for optimizer, color in colors.items():
        losses = results[optimizer]['test_losses']
        smoothed = smooth(losses, box_pts=3)
        ax.plot(smoothed, label=optimizer, linewidth=2.5, color=color, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Loss (Smoothed)', fontsize=12, fontweight='bold')
    ax.set_title('Smoothed Test Loss - Convergence Patterns', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_smoothing.png'), dpi=300, bbox_inches='tight')
    print("Saved: output/loss_smoothing.png")
    plt.close()


def generate_all_plots():
    """Generate all comparison plots."""
    print("\nGenerating visualizations...")
    print("-" * 50)
    results = load_results()
    
    plot_loss_comparison(results)
    plot_accuracy_comparison(results)
    plot_convergence_speed(results)
    plot_final_comparison(results)
    plot_loss_smoothing(results)
    
    print("-" * 50)
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
