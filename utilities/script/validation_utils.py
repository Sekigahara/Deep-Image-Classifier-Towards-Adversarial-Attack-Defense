import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

def get_training_plot(history, metric='accuracy', save=False, savepath="weight/"):
    if metric == 'loss':
        fig = plt.gcf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    elif metric == 'accuracy' or metric == 'acc':
        fig = plt.gcf()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    if save == True and (metric =='accuracy' or metric=='accuracy'):
        fig.savefig(savepath)
    elif save == True and metric=='loss':
        fig.savefig(savepath)

def get_confusion_matrix(y_true, y_pred, save=False, savepath="weight/"):
    print('\nConfusion Matrix\n')
    label_format = np.load("utilities/label_format.npz")['label']
    cm_result = confusion_matrix(y_true=y_true, y_pred=y_pred)
    #print(cm_result)

    df_cm = pd.DataFrame(cm_result, label_format, label_format)
    sns.set(font_scale=1.4)
    heatmap = sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='g')
    plt.show()

    print(df_cm)

    if save:
        fig = heatmap.get_figure()
        fig.savefig(savepath)

def get_classification_report(y_true, y_pred):
    label_format = np.load("utilities/label_format.npz")['label']
    print('\n\nClassification Report\n')
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=label_format))

def save_model(model, savepath="weight/", filename='model.h5'):
    model.save(savepath + filename)
