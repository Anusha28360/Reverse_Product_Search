from PIL import Image
import os

folder = "dataset/images"

for file in os.listdir(folder)[:5]:
    img = Image.open(os.path.join(folder, file))
    print(file, img.size)