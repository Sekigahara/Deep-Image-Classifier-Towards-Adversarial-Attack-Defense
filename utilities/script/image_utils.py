import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def image_loader(image_path):
    if image_path.split("/")[-1].split(".")[1].lower() == 'png':
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.divide(image, 255)

def show_image(image, label):
    plt.imshow(image)
    print("Corresponding label : {}".format(label))
