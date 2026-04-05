from pathlib import Path
from src.ImagePreprocessor import ImagePreprocessor

if __name__ == "__main__":
    path_image = "data/girls/best/5.jfif"
    ip = ImagePreprocessor()
    # ip.without_background2()
    ip.without_background(path_image)