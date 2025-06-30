def _accumulate_grad_handle_broadcasting(tensor, grad):
    """ Handle gradient accumulation with proper broadcasting. """
    if tensor.grad.shape == grad.shape:
        tensor.grad += grad
    else:
        # Handle broadcasting by summing over broadcasted dimensions
        # and reshaping to match the original tensor's shape
        grad_copy = grad.copy()

        # Sum over dimensions that were broadcasted
        ndims_added = grad.ndim - tensor.grad.ndim
        for _ in range(ndims_added):
            grad_copy = grad_copy.sum(axis=0)

        # Sum over dimensions that were size 1 in original tensor
        for i, (grad_dim, orig_dim) in enumerate(zip(grad_copy.shape, tensor.grad.shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad_copy = grad_copy.sum(axis=i, keepdims=True)

        tensor.grad += grad_copy