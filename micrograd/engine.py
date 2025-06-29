import numpy as np

def _accumulate_grad_handle_broadcasting(tensor, grad):
    if tensor.shape == ():  # scalar
        tensor.grad += np.sum(grad)
    else:
        # Handle broadcasting by summing over expanded dimensions
        if grad.shape != tensor.shape:
            # Find axes that were broadcasted
            axes = tuple(i for i, (a, b) in enumerate(zip(grad.shape, tensor.shape)) if a != b)
            grad = np.sum(grad, axis=axes, keepdims=True).reshape(tensor.shape)
        tensor.grad += grad

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data)
        self.grad = np.zeros(self.data.shape)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(value={self.data}, shape={self.shape}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        # Handle scalar addition by broadcasting
        try:
            out_data = self.data + other.data
        except ValueError as e:
            raise ValueError(f"Shape mismatch in addition: {self.data.shape} vs {other.data.shape}") from e
        out = Tensor(out_data, (self, other), "+")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, out.grad)
            _accumulate_grad_handle_broadcasting(other, out.grad)
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        
        def _backward():
            _accumulate_grad_handle_broadcasting(self, out.grad * other.data)
            _accumulate_grad_handle_broadcasting(other, out.grad * self.data)
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other), "**")
        
        def _backward():
            # da of a^b = b * a^(b-1) | Handle a=0 case by zeroing gradient
            base_grad = (out.grad * other.data * np.where(self.data != 0, self.data ** (other.data - 1), 0.0))
            _accumulate_grad_handle_broadcasting(self, base_grad)
            # db of a^b = a^b * ln(a) | Handle a<=0 case by zeroing gradient
            exp_grad = out.grad * np.where(self.data > 0, self.data**other.data * np.log(self.data), 0.0)
            _accumulate_grad_handle_broadcasting(other, exp_grad)
        out._backward = _backward
        
        return out

    def tanh(self):
        e = np.exp(2 * self.data)
        t = (e - 1) / (e + 1)
        out = Tensor(t, (self,), "TanH")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Tensor(np.max(0, self.data), (self,), "ReLU")
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "Exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        x = np.max(self.data, 1e-8)
        out = Tensor(np.log(x), (self,), "Log")
        
        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        
        return out
    
    def transpose(self):
        out = Tensor(self.data.T, (self,), "T")
        
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward

        return out

    @property
    def T(self):
        return self.transpose()
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Check matrix compatibility
        assert self.ndim != 0 and other.ndim != 0, "Scalars cannot be matrix-multiplied"
        if self.ndim == 1 and other.ndim == 1: # Dot product
            out_data = np.dot(self.data, other.data)
        else:
            try:
                out_data = self.data @ other.data
            except ValueError as e:
                raise ValueError(f"Shape mismatch in matmul: {self.shape} vs {other.shape}") from e

        out = Tensor(out_data, (self, other), "@")

        def _backward():
            if self.ndim == 1 and other.ndim == 1:  # Both vectors
                _accumulate_grad_handle_broadcasting(self, out.grad * other.data)
                _accumulate_grad_handle_broadcasting(other, out.grad * self.data)
            elif self.ndim == 1:  # Vector @ Matrix case
                _accumulate_grad_handle_broadcasting(self, (out.grad @ other.data.T).squeeze())
                _accumulate_grad_handle_broadcasting(other, np.outer(self.data, out.grad))
            elif other.ndim == 1:  # Matrix @ Vector case
                _accumulate_grad_handle_broadcasting(self, np.outer(out.grad, other.data))
                _accumulate_grad_handle_broadcasting(other, (self.data.T @ out.grad).squeeze())
            else:  # Matrix @ Matrix case
                _accumulate_grad_handle_broadcasting(self, out.grad @ other.data.T)
                _accumulate_grad_handle_broadcasting(other, self.data.T @ out.grad)

        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "Sigmoid")
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        # build topological graph of all the children
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()