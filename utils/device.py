import torch


def get_device():
    """
    Return the best available torch device.

    Priority:
    CUDA GPU → Apple MPS → CPU
    """

    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


def move_to_device(obj, device):
    """
    Safely move tensor or model to device.
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj
