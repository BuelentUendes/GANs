import os
from os.path import abspath

# Define the common paths
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = abspath(os.path.join(FILE_PATH, './../'))
MNIST_PATH = abspath(os.path.join(FILE_PATH, './../', 'MNIST_data'))