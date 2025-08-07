import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.module import Module, Linear, LayerNorm
from src.core.losses import CrossEntropy, BCE
from src.core.optim import Standard, AdamW
from src.core.tensor import Tensor
import numpy as np
import time
from typing import List
from src.tokenizer.tokenizer import Tokenizer
import pandas as pd


class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.linear(7, 10, name="fc1")

        self.fc2 = self.linear(10,1, name="fc2")
        self.ln = self.layer_norm(axis=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x, p=0.1)
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
    def train(self, x: Tensor, y: Tensor, optimizer, num_epochs=10000):
        for epoch in range(num_epochs):
            y_hat = self.forward(x)
            
            loss = BCE(y_hat, y)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 1 == 0:

                print(f"Epoch {epoch}, Loss: {loss.data}")

if __name__ == "__main__":
    df = pd.read_csv("src/experiments/data.csv")
    df['Quality'] = df['Quality'].apply(lambda x: 1 if x == "Good" else 0)
    X = Tensor(np.array(df.drop('Quality', axis=1).values))[:128]
    y = Tensor(np.array(df['Quality'].values).reshape((-1, 1)))[:128]

    X_test = Tensor(np.array(df.drop('Quality', axis=1).values))[128:]
    y_test = Tensor(np.array(df['Quality'].values).reshape((-1, 1)))[128:]

    net = Net()

    net._build(X.shape)
    optimizer = AdamW(net.parameters(), lr=0.001, clip_norm=10.0)

    net.train(X, y, optimizer)

    print(net.pipeline)
