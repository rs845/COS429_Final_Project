from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import os.path
import argparse
import cv2


def build_model(pretrained):
    model = Sequential([
        pretrained,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    model.summary()
    return model

def build_vgg(pretrained):
  model = Sequential([pretrained,
                    layers.Flatten(),
                    layers.Dense(2, activation = "sigmoid")])
  model.layers[0].trainable = False
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
  model.summary()
  return model


def train_dense(training_steps, validation_steps, train_flow, valid_flow):
    model = DenseNet121(
        weights= "imagenet",
        include_top=False,
        input_shape=(224,224,3)
    ) 
    densenet = build_model(model)
    densenet.summary()
    checkpoint_path = "results/cp.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    densenet.fit(
        train_flow,
        epochs = 1,
        steps_per_epoch = training_steps,
        validation_data = valid_flow,
        validation_steps = validation_steps,
        callbacks = [cp_callback]
    )
    return densenet

def train_inception(training_steps, validation_steps, train_flow, valid_flow):
    model = InceptionV3(
        weights= "imagenet",
        include_top=False,
        input_shape=(224,224,3)
    ) 
    inceptionnet = build_model(model)
    inceptionnet.summary()
    checkpoint_path = "results/inception.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    inc = inceptionnet.fit(
        train_flow,
        epochs = 1,
        steps_per_epoch = training_steps,
        validation_data = valid_flow,
        validation_steps = validation_steps,
        callbacks = [cp_callback]
    )
    return inceptionnet, inc

def train_xception(training_steps, validation_steps, train_flow, valid_flow):
    model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
        # classifier_activation="softmax",
    )
    Xceptionnet = build_model(model)
    Xceptionnet.summary()
    checkpoint_path = "results/Xception.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    Xceptionnet.fit(
        train_flow,
        epochs = 1,
        steps_per_epoch = training_steps,
        validation_data = valid_flow,
        validation_steps = validation_steps,
        callbacks = [cp_callback]
    )
    return Xceptionnet

def train_VGG(training_steps, validation_steps, train_flow, valid_flow):
    model = VGG19(
        include_top = False,
        weights="imagenet",
        input_shape=(224,224,3),
    )
    VGGmodel = build_vgg(model)
    
    checkpoint_path = "results/vgg.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    VGGmodel.fit(
        train_flow,
        epochs = 1,
        steps_per_epoch = training_steps,
        validation_data = valid_flow,
        validation_steps = validation_steps,
        callbacks = [cp_callback]
    )
    return model
    

MODEL_MAP = {
    "DenseNet" : train_dense,
    "InceptionNet" : train_inception,
    "XceptionNet" : train_xception,
    "VGG" : train_VGG
}

def train(model_name, train_flow, valid_flow):
    return MODEL_MAP[model_name](50000//64, 10000//64, train_flow, valid_flow)