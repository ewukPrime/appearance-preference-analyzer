from pathlib import Path
from src.ImagePreprocessor import ImagePreprocessor
import torch

if __name__ == "__main__":
    img_path = "data/girls/best/9.jpg"
    ip = ImagePreprocessor()
    # ip.without_background2()
    # ip.without_background(img_path)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    # ip.refine_mask_with_depth(img_path)