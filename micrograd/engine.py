import numpy as np
from micrograd import _accumulate_grad_handle_broadcasting

class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __len__(self):
        return len(self.data)
    
    def numel(self):
        return np.prod(self.shape)
    
    @property
    def item(self):
        return self.data.reshape(-1)[0] if self.numel() == 1 else self.data
    
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
        out = Tensor(self.data**other.data, (self, other), "**")

        def _backward():
            # da of a^b = b * a^(b-1) | Handle a=0 case by zeroing gradient
            base_grad = (out.grad * other.data * np.where(self.data != 0, self.data ** (other.data - 1), 0.0))
            _accumulate_grad_handle_broadcasting(self, base_grad)
            # db of a^b = a^b * ln(a) | Handle a<=0 case by zeroing gradient
            exp_grad = out.grad * np.where(self.data > 0, out.data * np.log(np.maximum(self.data, 1e-8)), 0.0)
            _accumulate_grad_handle_broadcasting(other, exp_grad)
        out._backward = _backward

        return out
    
    def transpose(self):
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, out.grad.T)
        out._backward = _backward

        return out

    @property
    def T(self):
        return self.transpose()

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        # Check matrix compatibility
        assert self.ndim != 0 and other.ndim != 0, "Scalars cannot be matrix-multiplied"
        if self.ndim == 1 and other.ndim == 1:  # Dot product
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
                grad_self = out.grad @ other.data.T
                if grad_self.ndim > 1:
                    grad_self = grad_self.squeeze()
                _accumulate_grad_handle_broadcasting(self, grad_self)
                _accumulate_grad_handle_broadcasting(other, np.outer(self.data, out.grad))
            elif other.ndim == 1:  # Matrix @ Vector case
                _accumulate_grad_handle_broadcasting(self, np.outer(out.grad, other.data))
                grad_other = self.data.T @ out.grad
                if grad_other.ndim > 1:
                    grad_other = grad_other.squeeze()
                _accumulate_grad_handle_broadcasting(other, grad_other)
            else:  # Matrix @ Matrix case
                _accumulate_grad_handle_broadcasting(self, out.grad @ other.data.T)
                _accumulate_grad_handle_broadcasting(other, self.data.T @ out.grad)
        out._backward = _backward
        
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "Sigmoid")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, out.data * (1 - out.data) * out.grad)
        out._backward = _backward
        
        return out
    
    def sum(self):
        out = Tensor(np.sum(self.data), (self,), "Sum")
        
        def _backward():
            # The gradient of sum is 1 for all input elements
            grad = np.ones_like(self.data) * out.grad
            _accumulate_grad_handle_broadcasting(self, grad)
        
        out._backward = _backward
        return out
    
    def __getitem__(self, key): # Indexing support
        out_data = self.data[key]
        out = Tensor(out_data, (self,), f"[{key}]")

        def _backward():
            grad_input = np.zeros_like(self.data)
            grad_input[key] = out.grad
            _accumulate_grad_handle_broadcasting(self, grad_input)
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

    def tanh(self):
        e = np.exp(2 * self.data)
        t = (e - 1) / (e + 1)
        out = Tensor(t, (self,), "TanH")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, (1 - t**2) * out.grad)
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), "ReLU")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, (out.data > 0) * out.grad)
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "Exp")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, out.data * out.grad)
        out._backward = _backward

        return out

    def log(self):
        x = np.maximum(self.data, 1e-8)
        out = Tensor(np.log(x), (self,), "Log")

        def _backward():
            _accumulate_grad_handle_broadcasting(self, (1 / x) * out.grad)
        out._backward = _backward

        return out
    
    def log_softmax(self):
        assert self.ndim <= 2 and (self.ndim == 1 or 1 in self.shape), "LogSoftmax requires 1D or 2D (1,n)/(n,1) tensor"
        
        # Numerically stable log-softmax
        x = self.data.reshape(-1)
        x_max = np.max(x)
        log_sum_exp = np.log(np.sum(np.exp(x - x_max))) + x_max
        out = Tensor((x - log_sum_exp).reshape(self.shape), (self,), "LogSoftmax")
        
        def _backward():
            # Gradient of log_softmax: dL/dx_i = dL/dy_i - softmax * sum(dL/dy_j)
            g = out.grad.reshape(-1)
            s = np.exp(out.data.reshape(-1))  # softmax values
            grad = g - s * g.sum()
            _accumulate_grad_handle_broadcasting(self, grad.reshape(self.shape))
            
        out._backward = _backward
        return out

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

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()