import torch


def select_device(requested_device: str) -> str:
    # Check if CUDA is requested and available
    if requested_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    # Check if MPS (Metal) is requested and available
    elif requested_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"  # Correct device name for MPS
        else:
            # Detailed check to provide specific reasons for MPS unavailability
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
    # Default to CPU if requested device is not available or if 'cpu' is requested
    return "cpu"
