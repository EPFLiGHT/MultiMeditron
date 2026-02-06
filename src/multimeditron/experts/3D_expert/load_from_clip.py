import numpy as np
import torch
from transformers import AutoModel


def load_model(model_name_or_path: str, device: torch.device = None, cache_dir: str = "/mloscratch/users/achahed/cache"):
    """
    Load a 3D CLIP model from the given path.
    
    Args:
        model_name_or_path: Path to the model checkpoint or HuggingFace model identifier
        device: Device to load the model on (defaults to CUDA if available)
        cache_dir: Directory to cache HuggingFace models
    
    Returns:
        The loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True  # Use cached files only, don't try to download
    )
    model = model.to(device=device)
    return model


def encode_img(model, image_path: str):
    """
    Encode a 3D medical image using the CLIP model.
    
    Args:
        model: The loaded CLIP model
        image_path: Path to the .npy image file or directory containing 3D_None.npy
        
    Returns:
        Image features tensor
        
    Notes:
        - The image shape needs to be processed as 1*32*256*256
        - The image needs to be normalized to 0-1 (Min-Max Normalization)
        - The image format needs to be .npy
    """
    import os
    import glob
    device = next(model.parameters()).device
    
    # Handle case where image_path is a directory
    if os.path.isdir(image_path):
        # Look for .npy files in the directory
        # Prefer 3D files first, then any .npy file
        npy_files = glob.glob(os.path.join(image_path, "*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {image_path}")
        
        # Try to find a 3D file first
        three_d_files = [f for f in npy_files if "3D" in os.path.basename(f)]
        if three_d_files:
            image_path = three_d_files[0]
        else:
            # Otherwise use the first .npy file
            image_path = npy_files[0]
    
    # Load the image
    image = np.load(image_path)
    
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    # Ensure proper shape: (batch, channels, depth, height, width)
    # Model expects img_size=[32, 256, 256] with in_channels=1
    if image.dim() == 3:
        # Shape is (D, H, W), add batch and channel dims
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 4:
        # Shape is (1, D, H, W), this is (batch, D, H, W), need to add channel dim
        # Or it could be (C, D, H, W) - check first dim
        if image.shape[0] == 1:
            # Likely (batch=1, D, H, W), need channel dim between batch and spatial
            image = image.unsqueeze(1)  # Now (batch, channels=1, D, H, W)
    
    image = image.to(device=device, dtype=torch.float32)
    
    with torch.inference_mode():
        image_features = model.encode_image(image)[:, 0]
    
    return image_features
