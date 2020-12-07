import pandas as pd
import time
import os
import os.path
import csv
import traceback
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

figures_dir = "../figures/results_plots/"
df = pd.read_csv('results/results.csv')

# Y axis: 3 models
# X axis: accuracy
# 4 bars per, for compressions (size 200)
def compression_accuracy():
    nearest_accuracy = df[(df["Image Compression"]=='nearest') & (df["Test Image Size"]==224)]["Test Accuracy"].tolist()
    box_accuracy = df[(df["Image Compression"]=='box') & (df["Test Image Size"]==224)]["Test Accuracy"].tolist()
    lanczos_accuracy = df[(df["Image Compression"]=='lanczos') & (df["Test Image Size"]==224)]["Test Accuracy"].tolist()
    hamming_accuracy = df[(df["Image Compression"]=='hamming') & (df["Test Image Size"]==224)]["Test Accuracy"].tolist()

    labels = ["DenseNet", 'InceptionNet', 'XceptionNet']
    xlabels = [" ", "DenseNet", " ", 'InceptionNet', " ", 'XceptionNet']
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35       # the width of the bars
    num_compressions = 4
    legend_labels = ["Nearest", "Box", "Lanczos", "Hamming"]

    results = [nearest_accuracy, box_accuracy, lanczos_accuracy, hamming_accuracy]

    fig, ax = plt.subplots()
    for i in range(num_compressions):
        results_subset = results[i]
        ax.bar(ind+(width*(i-num_compressions/2))/num_compressions, results_subset, width/num_compressions, alpha=0.5, label=legend_labels[i])
    ax.set_xticklabels(xlabels,fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.9,1.0])
    ax.set_title("Model Test Accuracies By Compression")
    ax.set_xlabel("Model")
    plt.legend(title="Compression Algorithm") #loc="center left", bbox_to_anchor=(1,0.5))
    #plt.show()


# Y axis: 3 models
# X axis: accuracy
# 3 bars per, for sizes (constant compression)
def size_accuracy():
    compression_type = 'hamming'
    num_sizes = 3

    accuracy_176 = df[(df["Image Compression"]==compression_type) & (df["Test Image Size"]==176)]["Test Accuracy"].tolist()
    accuracy_200 = df[(df["Image Compression"]==compression_type) & (df["Test Image Size"]==200)]["Test Accuracy"].tolist()
    accuracy_224 = df[(df["Image Compression"]==compression_type) & (df["Test Image Size"]==224)]["Test Accuracy"].tolist()

    labels = ["DenseNet", 'InceptionNet', 'XceptionNet']
    xlabels = [" ", "DenseNet", " ", 'InceptionNet', " ", 'XceptionNet']
    ind = np.arange(len(labels))  # the x locations for the groups
    width = 0.35       # the width of the bars
    num_compressions = 4
    legend_labels = ["176", "200", "224"]

    results = [accuracy_176, accuracy_200, accuracy_224]

    fig, ax = plt.subplots()
    for i in range(num_sizes):
        results_subset = results[i]
        ax.bar(ind+(width*(i-num_sizes/2))/num_sizes, results_subset, width/num_sizes, alpha=0.5, label=legend_labels[i])
    ax.set_xticklabels(xlabels,fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.6,1.0])
    ax.set_title(str("Model Test Accuracies For " + compression_type + " By Image Size"))
    ax.set_xlabel("Model")
    plt.legend(title="Test Image Size") #loc="center left", bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    #plt.show()

def show_images_size():
    # show example image at 256, 128, 64, 32
    path = 'display/HTKC64WWSG.jpg'

    im_256 = image.load_img(path, target_size=(256,256))
    im_128 = image.load_img(path, target_size=(128,128))
    im_64 = image.load_img(path, target_size=(64,64))
    im_32 = image.load_img(path, target_size=(32,32))

    im_256 = image.img_to_array(im_256)
    im_128 = image.img_to_array(im_128)
    im_64 = image.img_to_array(im_64)
    im_32 = image.img_to_array(im_32)
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(im_256/255.)
    axarr[0,1].imshow(im_128/255.)
    axarr[1,0].imshow(im_64/255.)
    axarr[1,1].imshow(im_32/255.)

    axarr[0,0].title.set_text('256x256')
    axarr[0,1].title.set_text('128x128')
    axarr[1,0].title.set_text('64x64')
    axarr[1,1].title.set_text('32x32')

    axarr[0,0].axes.get_xaxis().set_visible(False)
    axarr[0,0].axes.get_yaxis().set_visible(False)
    axarr[1,0].axes.get_xaxis().set_visible(False)
    axarr[1,0].axes.get_yaxis().set_visible(False)
    axarr[0,1].axes.get_xaxis().set_visible(False)
    axarr[0,1].axes.get_yaxis().set_visible(False)
    axarr[1,1].axes.get_xaxis().set_visible(False)
    axarr[1,1].axes.get_yaxis().set_visible(False)

    plt.show()


# show example image with 4 diff compressions + not compressed
def show_images_comp():
    # show example image at 256, 128, 64, 32
    path = 'display/I4AJHM4Y5N.jpg'

    im_nearest = image.load_img(path, target_size=(51,51), interpolation='nearest')
    im_box = image.load_img(path, target_size=(51,51), interpolation='box')
    im_lanczos = image.load_img(path, target_size=(51,51), interpolation='lanczos')
    im_hamming = image.load_img(path, target_size=(51,51), interpolation='hamming')

    im_nearest = image.img_to_array(im_nearest)
    im_box = image.img_to_array(im_box)
    im_lanczos = image.img_to_array(im_lanczos)
    im_hamming = image.img_to_array(im_hamming)
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(im_nearest/255., interpolation='nearest')
    axarr[0,1].imshow(im_box/255., interpolation='spline36')
    axarr[1,0].imshow(im_lanczos/255., interpolation='lanczos')
    axarr[1,1].imshow(im_hamming/255., interpolation='hamming')

    axarr[0,0].title.set_text('Nearest')
    axarr[0,1].title.set_text('Box')
    axarr[1,0].title.set_text('Lanczos')
    axarr[1,1].title.set_text('Hamming')

    axarr[0,0].axes.get_xaxis().set_visible(False)
    axarr[0,0].axes.get_yaxis().set_visible(False)
    axarr[1,0].axes.get_xaxis().set_visible(False)
    axarr[1,0].axes.get_yaxis().set_visible(False)
    axarr[0,1].axes.get_xaxis().set_visible(False)
    axarr[0,1].axes.get_yaxis().set_visible(False)
    axarr[1,1].axes.get_xaxis().set_visible(False)
    axarr[1,1].axes.get_yaxis().set_visible(False)

    plt.show()

compression_accuracy()
size_accuracy()
show_images_size()
show_images_comp()