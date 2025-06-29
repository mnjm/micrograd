# Source: https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
import torch
import numpy as np
from micrograd.engine import Tensor

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

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_broadcasting():
    a = Tensor([1.0, 2.0], [3.0, 4.0])
    b = Tensor([10.0, 20.0])
    c = a + b
    c.backward()
    print(a)
    print(b)
    print(c)

test_broadcasting()

# def test_broadcasting():
#     # Test broadcasting in operations
#     a = Tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = Tensor([10.0, 20.0])
#     c = a + b
#     d = c * b
#     d.backward()
#     amg, bmg, cmg, dmg = a, b, c, d

#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
#     b = torch.tensor([10.0, 20.0], requires_grad=True)
#     c = a + b
#     d = c * b
#     d.backward()

#     tol = 1e-6
#     assert np.allclose(dmg.data, d.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)
#     assert np.allclose(bmg.grad, b.grad.numpy(), tol)

# def test_transpose():
#     a = Tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = a.T
#     c = b @ a
#     c.backward()
#     amg, bmg, cmg = a, b, c

#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
#     b = a.T
#     c = b @ a
#     c.backward()

#     tol = 1e-6
#     assert np.allclose(cmg.data, c.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)

# def test_matmul():
#     # Test various matmul cases
#     # Matrix @ Matrix
#     a = Tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = Tensor([[5.0, 6.0], [7.0, 8.0]])
#     c = a @ b
#     c.backward()
#     amg, bmg, cmg = a, b, c

#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
#     b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
#     c = a @ b
#     c.backward()

#     tol = 1e-6
#     assert np.allclose(cmg.data, c.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)
#     assert np.allclose(bmg.grad, b.grad.numpy(), tol)

#     # Matrix @ Vector
#     a = Tensor([[1.0, 2.0], [3.0, 4.0]])
#     b = Tensor([5.0, 6.0])
#     c = a @ b
#     c.backward()
#     amg, bmg, cmg = a, b, c

#     a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
#     b = torch.tensor([5.0, 6.0], requires_grad=True)
#     c = a @ b
#     c.backward()

#     assert np.allclose(cmg.data, c.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)
#     assert np.allclose(bmg.grad, b.grad.numpy(), tol)

#     # Vector @ Vector (dot product)
#     a = Tensor([1.0, 2.0])
#     b = Tensor([3.0, 4.0])
#     c = a @ b
#     c.backward()
#     amg, bmg, cmg = a, b, c

#     a = torch.tensor([1.0, 2.0], requires_grad=True)
#     b = torch.tensor([3.0, 4.0], requires_grad=True)
#     c = a @ b
#     c.backward()

#     assert np.allclose(cmg.data, c.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)
#     assert np.allclose(bmg.grad, b.grad.numpy(), tol)


# def test_pow():
#     a = Tensor(2.0)
#     b = Tensor(3.0)
#     c = a**b
#     d = b**a
#     e = c + d
#     e.backward()
#     amg, bmg, cmg, dmg, emg = a, b, c, d, e

#     a = torch.tensor(2.0, requires_grad=True)
#     b = torch.tensor(3.0, requires_grad=True)
#     c = a**b
#     d = b**a
#     e = c + d
#     e.backward()

#     tol = 1e-6
#     assert abs(emg.data - e.detach().item()) < tol
#     assert abs(amg.grad - a.grad.item()) < tol
#     assert abs(bmg.grad - b.grad.item()) < tol


# def test_activations():
#     x = Tensor([-1.0, 0.0, 1.0])
#     y = x.tanh() + x.sigmoid() + x.relu()
#     y.backward()
#     xmg, ymg = x, y

#     x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
#     y = x.tanh() + x.sigmoid() + x.relu()
#     y.backward()

#     tol = 1e-6
#     assert np.allclose(ymg.data, y.detach().numpy(), tol)
#     assert np.allclose(xmg.grad, x.grad.numpy(), tol)


# def test_exp_log():
#     a = Tensor([0.1, 1.0, 2.0])
#     b = a.exp()
#     c = b.log()
#     c.backward()
#     amg, bmg, cmg = a, b, c

#     a = torch.tensor([0.1, 1.0, 2.0], requires_grad=True)
#     b = a.exp()
#     c = b.log()
#     c.backward()

#     tol = 1e-6
#     assert np.allclose(cmg.data, c.detach().numpy(), tol)
#     assert np.allclose(amg.grad, a.grad.numpy(), tol)


# def test_scalar_ops():
#     # Test operations with scalar tensors
#     a = Tensor(2.0)
#     b = Tensor(3.0)
#     c = a + b
#     d = a * b
#     e = a / b
#     f = a - b
#     g = a**b
#     h = (c + d + e + f + g).relu()
#     h.backward()
#     amg, bmg = a, b

#     a = torch.tensor(2.0, requires_grad=True)
#     b = torch.tensor(3.0, requires_grad=True)
#     c = a + b
#     d = a * b
#     e = a / b
#     f = a - b
#     g = a**b
#     h = (c + d + e + f + g).relu()
#     h.backward()

#     tol = 1e-6
#     assert abs(h.data - h.detach().item()) < tol
#     assert abs(amg.grad - a.grad.item()) < tol
#     assert abs(bmg.grad - b.grad.item()) < tol

# def test_inplace_ops():
#     a = Tensor(2.0)
#     b = Tensor(3.0)
#     a += b
#     a *= b
#     a -= b
#     a /= b
#     a.backward()
#     amg, bmg = a, b

#     a = torch.tensor(2.0, requires_grad=True)
#     b = torch.tensor(3.0, requires_grad=True)
#     a = a + b
#     a = a * b
#     a = a - b
#     a = a / b
#     a.backward()

#     tol = 1e-6
#     assert abs(amg.data - a.detach().item()) < tol
#     assert abs(amg.grad - a.grad.item()) < tol
#     assert abs(bmg.grad - b.grad.item()) < tol


# if __name__ == "__main__":
#     test_sanity_check()
#     test_more_ops()
#     test_broadcasting()
#     test_transpose()
#     test_matmul()
#     test_pow()
#     test_activations()
#     test_exp_log()
#     test_scalar_ops()
#     test_inplace_ops()
#     print("All tests passed!")