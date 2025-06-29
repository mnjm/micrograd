import numpy as np
from micrograd.engine import Tensor

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    
    def parameters(self):
        return []

class Layer(Module):
    def __init__(self, n_inp, n_out, act=None):
        scale = 1.0
        self.weights = Tensor(np.random.uniform(-scale, scale, (n_inp, n_out)))
        self.bias = Tensor(np.zeros((n_out)))
        self.act = act

    def __call__(self, x):
        out = (x @ self.weights) + self.bias
        assert (self.act in [None, 'relu', 'tanh', 'sigmoid']), "Unknown activation"
        if self.act == 'relu':
            out = out.relu()
        elif self.act == 'tanh':
            out = out.tanh()
        elif self.act == 'sigmoid':
            out = out.sigmoid()
        return out

    def parameters(self):
        return [ self.weights, self.bias ]
    
    def __repr__(self):
        return f"Layer({len(self.bias)}, act={self.act})"

class MultiLayerPerceptron(Module):
    def __init__(self, n_inp, n_outs):
        sz = [n_inp] + n_outs
        self.layers = [ Layer(sz[i], sz[i + 1], act=None if i==len(n_outs)-1 else 'relu') for i in range(len(n_outs)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"