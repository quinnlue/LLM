# DLX - Deep Learning eXperiments

A deep learning framework with custom tensor operations, neural network modules, and transformer implementations.

## Features

- **Custom Tensor Operations**: Implemented from scratch with automatic differentiation
- **Neural Network Modules**: Linear layers, Layer Normalization, and more
- **Transformer Implementation**: Complete transformer architecture with attention mechanisms
- **Optimizers**: AdamW, SGD, and other optimization algorithms
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy with logits
- **GPU Support**: Optional CUDA support via CuPy

## Installation

### From Source

```bash
git clone <repository-url>
cd transformer
pip install -e .
```

### With GPU Support

```bash
pip install -e .[gpu]
```

### Development Installation

```bash
pip install -e .[dev]
```

## Quick Start

```python
import dlx
from dlx import Tensor, Linear, CrossEntropyWithLogits, AdamW

# Create tensors
x = Tensor([[1, 2, 3], [4, 5, 6]])
y = Tensor([0, 1])

# Create a simple linear model
model = Linear(3, 2)
optimizer = AdamW(model.parameters(), lr=0.001)
criterion = CrossEntropyWithLogits()

# Forward pass
logits = model(x)
loss = criterion(logits, y)

# Backward pass
loss.backward()
optimizer.step()
```

## Package Structure

```
dlx/
├── nn/                 # Core neural network components
│   ├── tensor.py      # Custom tensor implementation
│   ├── module.py      # Base module class
│   ├── losses.py      # Loss functions
│   └── optim.py       # Optimizers
├── modules/           # Neural network layers
│   ├── linear.py      # Linear/fully connected layers
│   ├── layernorm.py   # Layer normalization
│   └── transformer.py # Transformer components
├── utils/             # Utility functions
│   ├── backend.py     # Backend management (numpy/cupy)
│   ├── logger.py      # Logging utilities
│   └── lr_scheduler.py # Learning rate scheduling
├── inference/         # Inference utilities
├── experiments/       # Experimental code and notebooks
└── tests/            # Test suite
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black dlx/
```

### Type Checking

```bash
mypy dlx/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
