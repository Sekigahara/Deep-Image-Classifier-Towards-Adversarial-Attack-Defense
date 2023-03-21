import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image
from collections import Counter
from sklearn.preprocessing import LabelEncoder

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

def dataset_ratio_check(y):
    counter = Counter(y)
    for k, v in counter.items():
        per= v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

def load_samples(size=(64, 64)):
    sample_images = []
    sample_ground_truth = []
    samples_list = glob.glob("sample/*")
    label_format = np.load("utilities/label_format.npz")['label']

    for idx, sample in enumerate(samples_list):
        ground_truth = sample.split("\\")[1].split("_")[0]

        sample_images.append(image_loader(sample, size))
        sample_ground_truth.append(ground_truth)

    le = LabelEncoder()
    le.fit(label_format)
    sample_ground_truth = le.transform(sample_ground_truth)

    return np.array(sample_images), np.array(sample_ground_truth)

def show_classification_result(images, adversaries, labels, adversary_labels):
    label_format = np.load("utilities/label_format.npz")['label']
    
    fig, ax = plt.subplots(len(images), 1)
    ax = ax.flatten()
    fig.set_figheight(30)
    fig.set_figwidth(30)

    output = []

    for idx, image in enumerate(images):
        color = (0, 255, 0)

        image_pred = labels[idx].argmax()
        adversary_pred = adversary_labels[idx].argmax()

        image_conf = max(labels[idx])
        adversary_conf = max(adversary_labels[idx])

        if image_pred != adversary_pred:
            color = (0, 0, 255)

        # Pui prediction
        cv2.putText(image, label_format[image_pred], (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(adversaries[idx], label_format[adversary_pred], (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Put confidence
        cv2.putText(image, str(image_conf), (2, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(adversaries[idx], str(adversary_conf), (2, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        output.append(np.hstack((image, adversaries[idx])))
        
    for idx, a in enumerate(ax):
        a.imshow(output[idx])
    
