import os

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from model import unet_model


TRAIN_DIR = 'data/train'
VAL_DIR = 'data/valid'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = (224, 224)


def load_image(image_path: str) -> tf.Tensor:
    image = Image.open(image_path)
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def load_mask(mask_path: str) -> tf.Tensor:
    mask = Image.open(mask_path)
    mask = np.array(mask)
    mask = mask[:, :, np.newaxis]
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    mask = tf.image.resize(mask, IMAGE_SIZE)
    mask = mask / 255.0
    return mask

def load_dataset(dataset_dir: str) -> (tf.Tensor, tf.Tensor):
    inputs, targets = [], []
    for image_path in os.listdir(os.path.join(dataset_dir, 'raw')):
        image = load_image(os.path.join(dataset_dir, 'raw', image_path))
        image = preprocess_image(image)
        inputs.append(image)

        target = load_mask(os.path.join(dataset_dir, 'mask', image_path))
        targets.append(target)

    inputs = tf.stack(inputs)
    targets = tf.stack(targets)
    return inputs, targets


val_inputs, val_targets = load_dataset(VAL_DIR)
inputs, targets = load_dataset(TRAIN_DIR)

model = unet_model()

model.compile(
    optimizer=tf.optimizers.Adam(LEARNING_RATE),
    loss=tf.losses.BinaryCrossentropy(),
    metrics=[tf.metrics.BinaryAccuracy()]
)

model.fit(
    x=inputs,
    y=targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_inputs, val_targets),
    shuffle=True
)

model.save('weights/model.h5')