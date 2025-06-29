import numpy as np

def accumulate_grad_handle_broadcasting(tensor, grad):
    """
    Accumulate gradients handling broadcasting correctly.
    When operations broadcast tensors, the gradients need to be "unbroadcast"
    by summing over the dimensions that were expanded.
    """
    if tensor.shape == ():  # scalar
        tensor.grad += np.sum(grad)
    else:
        if grad.shape != tensor.shape:
            # We need to reduce grad.shape to tensor.shape by summing appropriately

            # Handle the case where grad has more dimensions than tensor
            # This happens when tensor was implicitly prepended with size-1 dimensions
            ndim_diff = len(grad.shape) - len(tensor.shape)
            if ndim_diff > 0:
                # Sum over the leading dimensions
                axes_to_sum = tuple(range(ndim_diff))
                grad = np.sum(grad, axis=axes_to_sum)

            # Handle the case where corresponding dimensions have different sizes
            # This happens when tensor had size-1 dimensions that got broadcast
            axes_to_sum = []
            for i in range(len(tensor.shape)):
                if tensor.shape[i] == 1 and grad.shape[i] > 1:
                    axes_to_sum.append(i)

            if axes_to_sum:
                grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)

            # Ensure final shape matches
            grad = grad.reshape(tensor.shape)

        tensor.grad += grad