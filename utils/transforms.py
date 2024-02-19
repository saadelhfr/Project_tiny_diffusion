from torchvision import transforms
import torch


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
