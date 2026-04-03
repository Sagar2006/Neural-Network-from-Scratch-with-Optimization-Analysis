import numpy as np
from nn_from_scratch import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pickle
import os


def load_mnist():
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
    
    X = X / 255.0
    
    num_classes = len(np.unique(y))
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    
    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_one_hot[:train_size], y_one_hot[train_size:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_with_optimizer(optimizer_name, X_train, y_train, X_test, y_test, 
                         epochs=30, batch_size=128):
    """Train neural network with a specific optimizer."""
    print(f"\n{'='*50}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*50}")
    
    nn = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    num_batches = X_train.shape[0] // batch_size
    
    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        t = epoch + 1
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            predictions = nn.forward(X_batch)
            loss = nn.compute_loss(predictions, y_batch)
            epoch_loss += loss
            
            nn.backward(X_batch, y_batch)
            
            if optimizer_name == "SGD":
                nn.update_parameters_sgd()
            elif optimizer_name == "Momentum":
                nn.update_parameters_momentum(beta=0.9)
            elif optimizer_name == "RMSProp":
                nn.update_parameters_rmsprop(beta=0.999)
            elif optimizer_name == "Adam":
                nn.update_parameters_adam(t=t)
        
        train_loss = epoch_loss / num_batches
        test_predictions = nn.forward(X_test)
        test_loss = nn.compute_loss(test_predictions, y_test)
        train_acc = nn.accuracy(X_train[:5000], y_train[:5000])
        test_acc = nn.accuracy(X_test, y_test)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_test_accuracy': test_acc
    }


def main():
    """Main training loop comparing all optimizers."""
    X_train, y_train, X_test, y_test = load_mnist()
    
    results = {}
    optimizers = ["SGD", "Momentum", "RMSProp", "Adam"]
    
    for optimizer in optimizers:
        results[optimizer] = train_with_optimizer(
            optimizer, X_train, y_train, X_test, y_test,
            epochs=30, batch_size=128
        )
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for optimizer in optimizers:
        print(f"{optimizer:12} - Test Accuracy: {results[optimizer]['final_test_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()
