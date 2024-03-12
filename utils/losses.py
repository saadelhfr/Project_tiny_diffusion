import torch


def dispersion_loss(coordinates, epsilon=1e-8):
    """
    Calculate the dispersion loss for a batch of coordinates to discourage clustering.
    coordinates: Tensor of shape (batch_size, 2) representing x, y coordinates.
    epsilon: Small constant to avoid division by zero.
    """
    # Compute pairwise distance matrix
    dists = torch.cdist(coordinates, coordinates, p=2) + epsilon

    # Apply potential function, e.g., inverse square law, excluding self-pair distances (diagonal)
    inverse_square_dists = 1.0 / torch.square(dists)
    inverse_square_dists.fill_diagonal_(0)

    # Sum up all contributions and normalize by the number of pairs
    loss = torch.sum(inverse_square_dists) / (
        coordinates.shape[0] * (coordinates.shape[0] - 1)
    )
    return loss
