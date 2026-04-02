from pathlib import Path
from src.ImagePreprocessor import ImagePreprocessor

if __name__ == "__main__":
    path_image = "data/best/4.jfif"
    ip = ImagePreprocessor(path_image)
    # ip.without_background2()
    ip.without_background3(path_image)