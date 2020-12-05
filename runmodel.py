import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from models import train, build_model
import argparse
import pandas as pd
import time
import os
import os.path
import csv
import traceback
import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn import metrics
# import tensorflow as tf
# from tqdm import tqdm

COLUMNS = [
    "Model",
    "Image Compression",
    "Train Image Size",
    "Test Image Size",
    "Training Accuracy",
    "Test Accuracy",
    "True Negative",
    "False Positive",
    "False Negative",
    "True Positive",
    "Train Time",
    "Test Time"
]

# importing data
def dataFlow(train_size, test_size):
    base_path = '/Users/rachelsylwester/Desktop/archive/real_vs_fake/real-vs-fake/'
    image_gen = ImageDataGenerator(rescale=1./255.)

    train_flow = image_gen.flow_from_directory(
        base_path + 'train/',
        target_size=(train_size, train_size),
        batch_size=64,
        class_mode='binary'
    )

    valid_flow = image_gen.flow_from_directory(
        base_path + 'valid/',
        target_size=(train_size, train_size),
        batch_size=64,
        class_mode='binary',
        shuffle = False
        )

    test_flow_nearest = image_gen.flow_from_directory(
        base_path + 'test/',
        target_size=(test_size, test_size),
        batch_size=64,
        class_mode='binary',
        interpolation = "nearest",
        shuffle = False
    )

    test_flow_box = image_gen.flow_from_directory(
        base_path + 'test/',
        target_size=(test_size, test_size),
        batch_size=64,
        class_mode='binary',
        interpolation="box",
        shuffle = False
    )

    test_flow_lanczos = image_gen.flow_from_directory(
        base_path + 'test/',
        target_size=(test_size, test_size),
        batch_size=64,
        class_mode='binary',
        interpolation="lanczos",
        shuffle = False
    )

    test_flow_hamming = image_gen.flow_from_directory(
        base_path + 'test/',
        target_size=(test_size, test_size),
        batch_size=64,
        class_mode='binary',
        interpolation="hamming",
        shuffle = False
    )

    return train_flow, valid_flow, test_flow_nearest, test_flow_box, test_flow_lanczos, test_flow_hamming

def train_and_test(model_name, compression_type, training_size, test_size):
# train model
    if compression_type == "basic":
        start = time.process_time()
        print("training")
        model = train(model_name, train_flow, valid_flow)
        end = time.process_time()
    else:
        start = time.process_time()
        model = DenseNet121(
            weights= None,
            include_top=False,
            #input_shape=(224,224,3)
        ) 
        model = build_model(model)
        print("loading weight")
        model.load_weights("results/cp.ckpt")
        end = time.process_time()

    training_time = end-start
    

    FLOW_MAP = {
        "nearest": test_flow_nearest,
        "box": test_flow_box,
        "lanczos": test_flow_lanczos,
        "hamming": test_flow_hamming,
    }
    test_flow = FLOW_MAP[compression_type]

# save model
   # np.savez("model_%s.npz" % model_name, **model)

#Evaluate Model
    
    # get compression specific test flow
    # test_flow =

    start = time.process_time()
    train_pred = model.predict(valid_flow, steps=1)
    train_test = valid_flow.classes
    train_accuracy = metrics.accuracy_score(train_test[0:64], np.round(train_pred))
    # train_accuracy = history.history["accuracy"]
    test_pred = model.predict(test_flow, steps=1)
    test_test = test_flow.classes
    test_accuracy = metrics.accuracy_score(test_test[0:64], np.round(test_pred))
    print("Sample predictions: ", np.round(test_pred[0:64]))
    print("Sample actual labels: ", test_test[0:64])
    test_matrix = metrics.confusion_matrix(test_test[0:64], np.round(test_pred)).ravel()
    end = time.process_time()

    test_time = end-start


    # save results
    try:
        with open("results/%s_results.txt" % model_name, "a") as f:
            f.write("Model: %s\n" % model_name)
            f.write("Image Compression: %s\n" % compression)
            f.write("Train Image Size: %d\n" % training_size)
            f.write("Test Image Size: %d\n" % test_size)
            f.write("Train accuracy: %f\n" % train_accuracy)
            f.write("Test accuracy: %f\n" % test_accuracy)
            f.write("True Negatives: %f\n" % test_matrix[0])
            f.write("False Positives: %f\n" % test_matrix[1])
            f.write("False Negatives: %f\n" % test_matrix[2])
            f.write("True Positives: %f\n" % test_matrix[3])
            f.write("Training Time: %s sec\n" % training_time)            
            f.write("Testing Time: %s sec\n" % test_time)
            f.write("\n\n")
    except Exception as e:
        print(e)

    return {
        "Model" : model_name,
        "Image Compression" : compression,
        "Train Image Size" : training_size,
        "Test Image Size" :  test_size,
        "Training Accuracy" : train_accuracy,
        "Test Accuracy" : test_accuracy,
        "True Negative" : test_matrix[0],
        "False Positive" : test_matrix[1],
        "False Negative" : test_matrix[2],
        "True Positive": test_matrix[3],
        "Train Time": training_time,
        "Test Time" : test_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Only run experiments with this model")
    # parser.add_argument("--compression", help="Only run experiments with this compression")
    args = parser.parse_args()

    training_size = 224
    test_size = 128
    
    # initialize dataflow
    train_flow, valid_flow, test_flow_nearest, test_flow_box, test_flow_lanczos, test_flow_hamming = dataFlow(training_size, test_size)

    models = [
        "DenseNet",
        "InceptionNet",
        "XceptionNet",
        "VGG"
    ]

    if args.model not in models:
        raise Exception("Model not available")

    model_name = args.model

    compressions = [
        "nearest",
        "box",
        "lanczos",
        "hamming"
    ]

    # if args.compression not in compressions:
    #     raise Exception("Compression not available")

    # compression = args.compression

    save_path = "results/results.csv"

    for compression in compressions:
        # Create new save file if it doesn't exist
        if not os.path.exists(save_path):
            results = pd.DataFrame(columns=COLUMNS)
        else:
            results = pd.read_csv(save_path, float_precision='round_trip')

        try:
            results = results.append(train_and_test(model_name, compression, training_size, test_size), ignore_index=True)
            results.to_csv(save_path, index=False)
        except Exception as e:
            print(traceback.format_exc())
    


# print("Num training examples: %d" % num_train)
# save_path = "../results/results_%d.csv" % num_train

# # Create new save file if it doesn't exist
# if not os.path.exists(save_path):
#     results = pd.DataFrame(columns=COLUMNS)
# else:
#     results = pd.read_csv(save_path, float_precision='round_trip')







