from pathlib import Path
from src.ImagePreprocessor import ImagePreprocessor

if __name__ == "__main__":
    img_path = "data/girls/best/3.jfif"
    ip = ImagePreprocessor()
    # ip.without_background2()
    # ip.without_background(img_path)
    ip.refine_mask_with_depth(img_path)