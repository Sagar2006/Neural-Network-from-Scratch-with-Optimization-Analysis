import numpy as np


class NeuralNetwork:
    """Fully connected neural network built from scratch using NumPy."""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers specifying the size of each layer
                         (e.g., [784, 128, 64, 10] for MNIST)
            learning_rate: Learning rate for gradient updates
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = {}
        self.gradients = {}
        self.velocities = {}
        self.second_moments = {}
        self.m_t = {}
        self.v_t = {}
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights with He initialization and biases to zero."""
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] = np.random.randn(
                self.layer_sizes[i-1], 
                self.layer_sizes[i]
            ) * np.sqrt(2.0 / self.layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, self.layer_sizes[i]))
            self.velocities[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.velocities[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])
            self.m_t[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.m_t[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])
            self.v_t[f'W{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.v_t[f'b{i}'] = np.zeros_like(self.params[f'b{i}'])
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_dim)
        
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        self.cache = {}
        A = X
        
        for i in range(1, len(self.layer_sizes)):
            self.cache[f'A{i-1}'] = A
            Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
            self.cache[f'Z{i}'] = Z
            
            if i == len(self.layer_sizes) - 1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
        
        self.cache[f'A{len(self.layer_sizes)-1}'] = A
        return A
    
    def backward(self, X, y, learning_rate=None):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data
            y: Target labels (one-hot encoded)
            learning_rate: Optional learning rate override
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        m = X.shape[0]
        num_layers = len(self.layer_sizes)
        
        dA = self.cache[f'A{num_layers-1}'] - y
        
        for i in range(num_layers - 1, 0, -1):
            dZ = dA
            if i > 1:
                dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])
            
            dW = np.dot(self.cache[f'A{i-1}'].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            self.gradients[f'W{i}'] = dW
            self.gradients[f'b{i}'] = db
            
            if i > 1:
                dA = np.dot(dZ, self.params[f'W{i}'].T)
    
    def update_parameters_sgd(self):
        """Standard SGD parameter update."""
        for i in range(1, len(self.layer_sizes)):
            self.params[f'W{i}'] -= self.learning_rate * self.gradients[f'W{i}']
            self.params[f'b{i}'] -= self.learning_rate * self.gradients[f'b{i}']
    
    def update_parameters_momentum(self, beta=0.9):
        """Update parameters using Momentum optimizer."""
        for i in range(1, len(self.layer_sizes)):
            self.velocities[f'W{i}'] = (beta * self.velocities[f'W{i}'] - 
                                       self.learning_rate * self.gradients[f'W{i}'])
            self.velocities[f'b{i}'] = (beta * self.velocities[f'b{i}'] - 
                                       self.learning_rate * self.gradients[f'b{i}'])
            
            self.params[f'W{i}'] += self.velocities[f'W{i}']
            self.params[f'b{i}'] += self.velocities[f'b{i}']
    
    def update_parameters_rmsprop(self, beta=0.999, epsilon=1e-8):
        """Update parameters using RMSProp optimizer."""
        for i in range(1, len(self.layer_sizes)):
            self.v_t[f'W{i}'] = (beta * self.v_t[f'W{i}'] + 
                                (1 - beta) * self.gradients[f'W{i}']**2)
            self.v_t[f'b{i}'] = (beta * self.v_t[f'b{i}'] + 
                                (1 - beta) * self.gradients[f'b{i}']**2)
            
            self.params[f'W{i}'] -= (self.learning_rate * self.gradients[f'W{i}'] / 
                                    (np.sqrt(self.v_t[f'W{i}']) + epsilon))
            self.params[f'b{i}'] -= (self.learning_rate * self.gradients[f'b{i}'] / 
                                    (np.sqrt(self.v_t[f'b{i}']) + epsilon))
    
    def update_parameters_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """
        Update parameters using Adam optimizer.
        
        Args:
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            t: Timestep for bias correction
        """
        for i in range(1, len(self.layer_sizes)):
            self.m_t[f'W{i}'] = beta1 * self.m_t[f'W{i}'] + (1 - beta1) * self.gradients[f'W{i}']
            self.m_t[f'b{i}'] = beta1 * self.m_t[f'b{i}'] + (1 - beta1) * self.gradients[f'b{i}']
            
            self.v_t[f'W{i}'] = beta2 * self.v_t[f'W{i}'] + (1 - beta2) * self.gradients[f'W{i}']**2
            self.v_t[f'b{i}'] = beta2 * self.v_t[f'b{i}'] + (1 - beta2) * self.gradients[f'b{i}']**2
            
            m_hat_W = self.m_t[f'W{i}'] / (1 - beta1**t)
            m_hat_b = self.m_t[f'b{i}'] / (1 - beta1**t)
            v_hat_W = self.v_t[f'W{i}'] / (1 - beta2**t)
            v_hat_b = self.v_t[f'b{i}'] / (1 - beta2**t)
            
            self.params[f'W{i}'] -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
            self.params[f'b{i}'] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
    
    def compute_loss(self, predictions, y):
        """Compute categorical cross-entropy loss."""
        m = y.shape[0]
        log_probs = -np.log(predictions[range(m), np.argmax(y, axis=1)] + 1e-8)
        return np.mean(log_probs)
    
    def predict(self, X):
        """Make predictions on input data."""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def accuracy(self, X, y):
        """Calculate accuracy on given data."""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)
