from ImagePreprocessor import ImagePreprocessor
from pathlib import Path

if __name__ == "__main__":
    path_image = Path(f"data/best/1.jfif")
    ip = ImagePreprocessor(path_image)
    ip.without_background()
