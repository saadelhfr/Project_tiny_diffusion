import torch


def custom_cumprod(tensor, dim=0):
    """
    Manually implement the cumulative product.

    Args:
    - tensor (torch.Tensor): Input tensor for which to compute the cumulative product.
    - dim (int): The dimension over which to compute the cumulative product.

    Returns:
    - torch.Tensor: Tensor with the same shape as input, containing the cumulative products.
    """
    # Initialize the output tensor with the same shape and type as the input
    cumprod = torch.empty_like(tensor)

    # Start the cumulative product with 1
    cumprod_value = 1
    for i in range(tensor.size(dim)):
        cumprod_value *= tensor.select(dim, i)
        cumprod.select(dim, i).copy_(cumprod_value)

    return cumprod
