import os
import urllib.request
import gzip
import shutil
import numpy as np
import struct
from array import array
import tensorflow as tf
import tensorflow_datasets as tfds


def g_unzip(gz_in, out):
    with gzip.open(gz_in, 'rb') as f_in:
        with open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def reshape_grayscale_to_rgb(images: np.ndarray):
    last_dim = np.shape(images)[-1]
    if last_dim != 1:
        images = tf.convert_to_tensor(
            # e.g. (1500, 28, 28) --> (1500, 28, 28, 1)))
            images.reshape(tuple(list(np.shape(images)) + [1])))  
    else:
        images = tf.convert_to_tensor(images)
     # e.g. (1500, 28, 28, 1) --> (1500, 28, 28, 3)
    images_rgb = tf.image.grayscale_to_rgb(images) 
    return np.array(images_rgb)


def download_mnist_dataset():
    dataset = 'datasets/mnist/'
    os.makedirs(dataset, exist_ok=True)
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    for gz in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
    ]:
        url = f'{base_url}{gz}'
        print(f'Getting {url}...')
        urllib.request.urlretrieve(url, os.path.join(os.path.dirname(__file__), dataset + gz))
        g_unzip(dataset + gz, dataset + os.path.splitext(gz)[0])
    mnist_dl = MnistDataloader(
        training_images_filepath='datasets/mnist/train-images-idx3-ubyte',
        training_labels_filepath='datasets/mnist/train-labels-idx1-ubyte',
        test_images_filepath='datasets/mnist/t10k-images-idx3-ubyte',
        test_labels_filepath='datasets/mnist/t10k-labels-idx1-ubyte'
    )
    x_train, *_ = mnist_dl.load_data()
    x_train = reshape_grayscale_to_rgb(x_train)
    np.save('datasets/mnist/mnist.npy', x_train)


def resize_and_rescale(img, size):
    """Resize the image to the desired size first and then
    rescale the pixel values in the range [-1.0, 1.0].

    Args:
        img: Image tensor
        size: Desired image size for resizing (width * heigth)
    Returns:
        Resized and rescaled image tensor
    """

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)

    img = tf.image.crop_to_bounding_box(
        img,
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # Resize
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)
    # Rescale the pixel values
    img = img / 127.5 - 1.0
    img = tf.clip_by_value(img, -1.0, 1.0)
    return img


def download_tensorflow_dataset(dataset_name: str, max_imgs: None| int = None):
    if max_imgs == None:
        split = 'train'
    else:
        split = f"train[:{max_imgs}]"
    os.makedirs(f'datasets/tensorflow', exist_ok=True)
    (ds,) = tfds.load(dataset_name, split=[split], with_info=False, shuffle_files=True)
    ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
    images = []
    x_size_min, y_size_min = np.inf, np.inf
    for image_dict in ds_numpy:
      # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
        x_size, y_size = np.shape(image_dict['image'])[0], np.shape(image_dict['image'])[1]
        if x_size < x_size_min:
            x_size_min = x_size
        if y_size < y_size_min:
            y_size_min = y_size
        images.append(image_dict['image'])
    resized_images = []
    for image in images:
        resized_images.append(resize_and_rescale(image, size=(x_size_min, y_size_min)))
    np.save(f'datasets/tensorflow/{dataset_name}.npy', np.array(resized_images))


class MnistDataloader(object):
    """https://www.kaggle.com/code/hojjatk/read-mnist-dataset"""
    def __init__(
            self, 
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath, 
            test_labels_filepath
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        return np.array(images), labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return x_train, y_train, x_test, y_test        


if __name__ == '__main__':
    download_mnist_dataset()
    download_tensorflow_dataset("oxford_flowers102")
    download_tensorflow_dataset("food101", max_imgs=10000)
