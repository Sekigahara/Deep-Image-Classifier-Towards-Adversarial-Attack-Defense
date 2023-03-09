import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def image_loader(image_path, resize=(64, 64)):
    if image_path.split("/")[-1].split(".")[1].lower() == 'png':
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.divide(cv2.resize(image, resize), 255)

def show_image(image, label):
    plt.imshow(image)
    print("Corresponding label : {}".format(label))

def show_image_number(image, label):
    label_format = np.load("utilities/label_format.npz")['label']
    print(label_format)

    plt.imshow(image)
    print("Corresponding label : {}".format(label_format[label]))


def load_numpy_dataset(train_path, test_path, k):
    x_train, y_train = train_path["x"], train_path["y"]
    x_test, y_test = test_path["x"], test_path["y"]

    return (x_train, y_train), (x_test, y_test)
