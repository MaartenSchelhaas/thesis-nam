import torch


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Returns:
        torch.device: 'cuda' or 'cpu'
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
