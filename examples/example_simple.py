"""
Minimal example demonstrating the neural network on a small synthetic dataset.
This shows the basic usage without requiring full MNIST training.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from nn_from_scratch import NeuralNetwork


def create_synthetic_data(n_samples=1000):
    """Create a simple 2-class classification dataset."""
    np.random.seed(42)
    
    X_0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    y_0 = np.zeros((n_samples // 2, 1))
    
    X_1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    y_1 = np.ones((n_samples // 2, 1))
    
    X = np.vstack([X_0, X_1])
    y = np.vstack([y_0, y_1])
    
    y_one_hot = np.hstack([1 - y, y])
    
    return X, y_one_hot


def train_simple_example():
    """Train on synthetic data to demonstrate the network."""
    print("Generating synthetic dataset...")
    X, y = create_synthetic_data(1000)
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    print("Initializing network...")
    nn = NeuralNetwork([2, 16, 8, 2], learning_rate=0.01)
    
    print("\nTraining on synthetic data...")
    train_losses = []
    train_accs = []
    
    for epoch in range(100):
        predictions = nn.forward(X)
        loss = nn.compute_loss(predictions, y)
        train_losses.append(loss)
        
        nn.backward(X, y)
        nn.update_parameters_adam(t=epoch + 1)
        
        acc = nn.accuracy(X, y)
        train_accs.append(acc)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    
    print("\nTraining complete!")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, linewidth=2, color='#d62728')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, linewidth=2, color='#2ca02c')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'synthetic_example.png'), dpi=150)
    print("Saved visualization to output/synthetic_example.png")
    plt.close()


if __name__ == "__main__":
    print("="*50)
    print("Simple Example - Classification on Synthetic Data")
    print("="*50)
    train_simple_example()
