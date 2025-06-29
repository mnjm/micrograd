import torch
import numpy as np
from micrograd.engine import Tensor

tol = 1e-6  # float tollerence

def test_broadcasting():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([10.0, 20.0])
    c = a + b
    d = c * b
    d.backward()
    amg, bmg, dmg = a, b, d

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([10.0, 20.0], dtype=torch.double, requires_grad=True)
    c = a + b
    d = c * b
    d.sum().backward()

    assert np.allclose(dmg.data, d.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    assert np.allclose(bmg.grad, b.grad.numpy(), tol)

def test_transpose():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.T
    c = b @ a
    c.backward()
    amg, cmg = a, c

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double, requires_grad=True)
    b = a.T
    c = b @ a
    c.sum().backward()

    assert np.allclose(cmg.data, c.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)

def test_matmul():
    # 1. Matrix @ Matrix
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    c.backward()
    amg, bmg, cmg = a, b, c

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.double, requires_grad=True)
    c = a @ b
    c.sum().backward()

    assert np.allclose(cmg.data, c.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    assert np.allclose(bmg.grad, b.grad.numpy(), tol)

    # 2. Matrix @ Vector
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([5.0, 6.0])
    c = a @ b
    c.backward()
    amg, bmg, cmg = a, b, c

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double, requires_grad=True)
    b = torch.tensor([5.0, 6.0], dtype=torch.double, requires_grad=True)
    c = a @ b
    c.sum().backward()

    assert np.allclose(cmg.data, c.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    assert np.allclose(bmg.grad, b.grad.numpy(), tol)

    # 3. Vector @ Vector (dot product)
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a @ b
    c.backward()
    amg, bmg, cmg = a, b, c

    a = torch.tensor([1.0, 2.0], dtype=torch.double, requires_grad=True)
    b = torch.tensor([3.0, 4.0], dtype=torch.double, requires_grad=True)
    c = a @ b
    c.sum().backward()

    assert np.allclose(cmg.data, c.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    assert np.allclose(bmg.grad, b.grad.numpy(), tol)

def test_pow():
    a = Tensor([1.0, 2.0, 4.0])
    b = Tensor(3.0)
    c = a**b
    d = b**a
    e = c + d
    e.backward()
    amg, bmg, emg = a, b, e

    a = torch.tensor([1.0, 2.0, 4.0], dtype=torch.double, requires_grad=True)
    b = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    c = a**b
    d = b**a
    e = c + d
    e.sum().backward()

    assert np.allclose(emg.data, e.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    assert np.allclose(bmg.grad, b.grad.numpy(), tol)

def test_activations():
    x = Tensor([-1.0, 0.0, 1.0])
    y = x.tanh() + x.sigmoid() + x.relu()
    y.backward()
    xmg, ymg = x, y

    x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.double, requires_grad=True)
    y = x.tanh() + x.sigmoid() + x.relu()
    y.sum().backward()

    assert np.allclose(ymg.data, y.detach().numpy(), tol)
    assert np.allclose(xmg.grad, x.grad.numpy(), tol)

def test_exp_log():
    a = Tensor([0.1, 1.0, 2.0])
    b = a.exp()
    c = b.log()
    c.backward()
    amg, cmg = a, c

    a = torch.tensor([0.1, 1.0, 2.0], dtype=torch.double, requires_grad=True)
    b = a.exp()
    c = b.log()
    c.sum().backward()

    assert np.allclose(cmg.data, c.detach().numpy(), tol)
    assert np.allclose(amg.grad, a.grad.numpy(), tol)
    
# Adapted from - https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
def test_sanity_check():
    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():
    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

if __name__ == "__main__":
    import traceback

    test_list = [
        test_sanity_check,
        test_more_ops,
        test_broadcasting,
        test_transpose,
        test_matmul,
        test_pow,
        test_activations,
        test_exp_log,
    ]

    for test in test_list:
        try:
            test()
        except:
            print(f"Failed - {test.__name__}")
            traceback.print_exc()
            break
    else:
        print("All test passed")