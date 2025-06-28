import random
from micrograd.engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_inp, nonlin='relu'):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_inp)]
        self.bias = Value(0) # Bias should be init'd to 0
        self.nonlin = nonlin

    def __call__(self, x):
        # w * x + b
        out = sum([wi * xi for wi, xi in zip(self.weights, x)], self.bias)
        if self.nonlin:
            if self.nonlin == "tanh":
                out = out.tanh()
            elif self.nonlin == "relu":
                out = out.relu()
        return out

    def parameters(self):
        return self.weights + [ self.bias ]
    
    def __repr__(self):
        return f"Neuron{'+' + self.nonlin if self.nonlin else ''}({len(self.weights)})"

class Layer(Module):
    def __init__(self, n_inp, n_out, **kwargs):
        self.neurons = [ Neuron(n_inp, **kwargs) for _ in range(n_out) ]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [ p for neuron in self.neurons for p in neuron.parameters() ]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MultiLayerPerceptron(Module):
    def __init__(self, n_inp, n_outs):
        sz = [n_inp] + n_outs
        self.layers = [ Layer(sz[i], sz[i + 1], nonlin=None if i==len(n_outs)-1 else 'relu') for i in range(len(n_outs)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [ p for layer in self.layers for neuron in layer.neurons for p in neuron.parameters() ]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"