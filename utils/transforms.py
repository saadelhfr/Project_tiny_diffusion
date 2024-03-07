from torchvision import transforms
import torch


class ImageToNormalizedCoordinatesTransform:
    def __init__(self, keep_non_zero_only=True):
        self.keep_non_zero_only = keep_non_zero_only

    def __call__(self, image_tensor):
        if image_tensor.dim() == 3 and image_tensor.size(0) == 1:
            image_tensor = image_tensor.squeeze(0)  # Reduces to [H, W]
        elif image_tensor.dim() not in [2, 3]:
            raise ValueError(
                "image_tensor must be 2D or 3D with a singleton channel dimension"
            )

        # Assuming image_tensor is a 2D tensor of shape [H, W]
        H, W = image_tensor.shape
        result = []

        for y in range(H):
            for x in range(W):
                intensity = image_tensor[y, x]
                # Check if we need to keep this pixel based on its intensity
                if self.keep_non_zero_only and intensity == 0:
                    continue  # Skip this pixel if we are keeping non-zero pixels only and this pixel's intensity is 0

                # Normalize coordinates to be in the range [0, 1]
                x_normalized = x / (W - 1)
                y_normalized = y / (H - 1)
                result.append((x_normalized, y_normalized, intensity.item()))

        # Convert the result list to a tensor
        result_tensor = torch.tensor(result, dtype=torch.float32)
        return result_tensor


class MNISTToFlattenedTransform:
    def __call__(self, image):
        # Convert the PIL Image to a Tensor and remove the channel dimension
        image_tensor = transforms.ToTensor()(image).squeeze(0)

        # Flatten the image
        flattened = image_tensor.flatten().unsqueeze(-1)

        return flattened


class InverseFlattenedToMNISTTransform:
    def __call__(self, data):
        # Assuming data is a flat tensor of shape (784, 1)
        # Reshape it back to the original image size (28, 28)
        image = data.reshape(28, 28)
        return image
