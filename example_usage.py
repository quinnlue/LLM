#!/usr/bin/env python3
"""
Example usage of the DLX package.

This script demonstrates how to use the core components of the DLX package
including tensors, modules, losses, and optimizers.
"""

import dlx
from dlx import Tensor, Module, Linear, CrossEntropyWithLogits, AdamW

def main():
    print("DLX Package Example")
    print("=" * 50)
    
    # Create some sample data
    print("1. Creating tensors...")
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = Tensor([0, 1])  # Target labels
    
    print(f"Input tensor shape: {x.data.shape}")
    print(f"Target tensor shape: {y.data.shape}")
    
    # Create a simple linear model
    print("\n2. Creating a linear model...")
    model = Module()
    linear_layer = model.linear(3, 2, use_bias=True)
    print(f"Model parameters: {len(model.parameters())}")
    
    # Create optimizer and loss function
    print("\n3. Setting up optimizer and loss...")
    optimizer = AdamW(model.parameters(), lr=0.001)
    
    # Forward pass
    print("\n4. Running forward pass...")
    logits = linear_layer(x)
    loss = CrossEntropyWithLogits(logits, y)
    
    print(f"Logits shape: {logits.data.shape}")
    print(f"Loss value: {loss.data}")
    
    # Backward pass
    print("\n5. Running backward pass...")
    loss.backward()
    
    # Check gradients
    print("\n6. Checking gradients...")
    for name, param in model.parameters().items():
        if param.grad is not None:
            print(f"{name}: grad shape = {param.grad.data.shape}")
    
    # Optimizer step
    print("\n7. Taking optimizer step...")
    optimizer.step()
    
    print("\nExample completed successfully!")
    print("The DLX package is working correctly!")

if __name__ == "__main__":
    main()
