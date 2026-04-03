# Mathematical Foundation & Implementation Guide

## 1. Forward Propagation

### Layer-wise Computation
For each layer $i$, we compute:

**Pre-activation (Linear transformation):**
$$Z^{(i)} = A^{(i-1)} \cdot W^{(i)} + b^{(i)}$$

**Activation:**
$$A^{(i)} = \sigma(Z^{(i)})$$

Where:
- $A^{(i)}$: Activated output of layer $i$
- $W^{(i)}$: Weight matrix of layer $i$
- $b^{(i)}$: Bias vector of layer $i$
- $\sigma$: Activation function (ReLU for hidden layers, Softmax for output)

### Activation Functions

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(x) = \max(0, x)$$

**Softmax (for multi-class classification):**
$$\text{Softmax}(z_j) = \frac{e^{z_j}}{\sum_{k} e^{z_k}}$$

## 2. Loss Function

**Categorical Cross-Entropy:**
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$$

Where:
- $m$: Number of samples
- $C$: Number of classes
- $y_{ij}$: True label (one-hot encoded)
- $\hat{y}_{ij}$: Predicted probability

## 3. Backward Propagation

### Output Layer Gradient
For softmax + cross-entropy:
$$\frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \hat{Y} - Y$$

Where $L$ is the output layer, $Y$ is one-hot encoded labels, and $\hat{Y}$ is predictions.

### Hidden Layer Gradients
Backpropagation through ReLU:
$$\frac{\partial \mathcal{L}}{\partial Z^{(i)}} = \left(\frac{\partial \mathcal{L}}{\partial A^{(i)}} \cdot \text{ReLU}'(Z^{(i)})\right)$$

Where:
$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### Weight and Bias Gradients
$$\frac{\partial \mathcal{L}}{\partial W^{(i)}} = \frac{1}{m} (A^{(i-1)})^T \cdot \frac{\partial \mathcal{L}}{\partial Z^{(i)}}$$

$$\frac{\partial \mathcal{L}}{\partial b^{(i)}} = \frac{1}{m} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z^{(i)}_j}$$

## 4. Optimization Algorithms

### SGD (Stochastic Gradient Descent)
**Update rule:**
$$\theta_{t+1} = \theta_t - \alpha \nabla \mathcal{L}(\theta_t)$$

**Pros:** Simple, low memory  
**Cons:** High variance, may oscillate

### Momentum
**Maintains velocity:**
$$v_t = \beta v_{t-1} - \alpha \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_t$$

**Common settings:** $\beta = 0.9$

**Intuition:** Accumulate gradient history like a rolling ball gaining momentum

### RMSProp (Root Mean Square Propagation)
**Adapts learning rate per parameter:**
$$v_t = \beta v_{t-1} + (1-\beta) (\nabla \mathcal{L}(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t + \epsilon}} \nabla \mathcal{L}(\theta_t)$$

**Common settings:** $\beta = 0.999$, $\epsilon = 10^{-8}$

**Intuition:** Divide by RMS of gradient history to normalize learning rates

### Adam (Adaptive Moment Estimation)
**Combines first and second moments with bias correction:**

First moment (mean):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}(\theta_t)$$

Second moment (uncentered variance):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L}(\theta_t))^2$$

Bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Update:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t + \epsilon}} \hat{m}_t$$

**Common settings:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

## 5. Weight Initialization

### He Initialization (for ReLU)
$$W^{(i)} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

Where $n_{in}$ is the number of input units.

**Why:** Maintains similar activation variance across layers, preventing vanishing/exploding gradients.

## 6. Implementation Details

### Numerical Stability in Softmax
To prevent overflow:
$$\text{Softmax}(z_j) = \frac{e^{z_j - \max(z)}}{\sum_k e^{z_k - \max(z)}}$$

### Mini-batch Training
- Process samples in batches to:
  - Reduce computational cost
  - Add regularization (stochastic noise)
  - Improve generalization
  - Stabilize gradient estimates

### Gradient Accumulation
$$\nabla \mathcal{L}_{\text{batch}} = \frac{1}{|batch|} \sum_{i \in batch} \nabla \mathcal{L}(x_i)$$

## 7. Model Capacity

**Number of parameters:**
- Layer 1: $784 \times 128 + 128 = 100,480$
- Layer 2: $128 \times 64 + 64 = 8,256$
- Layer 3: $64 \times 10 + 10 = 650$
- **Total:** ~109,000 parameters

**Why sufficient for MNIST:**
- MNIST is a relatively simple dataset (hand-written digits)
- High model capacity allows fitting training data easily
- Dropout/regularization could improve beyond 99% accuracy

## 8. Convergence Analysis

### Learning Rate Considerations
- **Too high**: May diverge or oscillate around minimum
- **Too low**: Very slow convergence
- **Optimal range** for MNIST: 0.01 - 0.1
- **Adam's advantage**: Adapts learning rate per parameter

### Convergence Metrics
1. **Loss reduction**: Decreasing training loss indicates learning
2. **Accuracy improvement**: Primary metric for classification
3. **Generalization gap**: Difference between train and test accuracy
4. **Training stability**: Smooth curves indicate stable learning

## 9. Why Manual Implementation?

| Aspect | Manual | Framework |
|--------|--------|-----------|
| **Understanding** | Deep | Limited |
| **Debugging** | Easier | Harder |
| **Flexibility** | High | Lower |
| **Performance** | Slower | Optimized |
| **Educational Value** | Excellent | Students miss details |

## 10. Extensions & Improvements

### To achieve higher accuracy (99%+):
1. **Batch Normalization**: Normalize layer inputs
2. **Dropout**: Regularization to prevent overfitting
3. **Learning rate scheduling**: Reduce $\alpha$ over time
4. **Deeper networks**: Add more layers
5. **Data augmentation**: Rotation, shifting of digits

### Example batch norm update:
$$\hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y^{(i)} = \gamma \hat{x}^{(i)} + \beta$$

---

## References

1. **Kingma & Ba (2015)**: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
2. **Tieleman & Hinton (2012)**: [RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
3. **He et al. (2015)**: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852)
4. **LeCun et al. (1998)**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
5. **Rumelhart et al. (1986)**: [Learning Representations by Back-Propagating Errors](https://www.nature.com/articles/323533a0)
