import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
import time


LABELS = [
    "box",
    "clap",
    "Jog",
    "sit",
    "stand",
    "walk",
    "wave"
]
DATASET_PATH = "/home/yash/ARdata/DRONESDatasetExperiments/"

X_train_path = DATASET_PATH + "X_Train.txt"
X_test_path = DATASET_PATH + "X_Test.txt"

y_train_path = DATASET_PATH + "Y_Train.txt"
y_test_path = DATASET_PATH + "Y_Test.txt"

n_steps = 48 # 48 timesteps per series


# Load the networks inputs

def load_X(X_path):
    file = open(X_path, 'r')
    X_ = np.array(
        [elem for elem in [
            row.split(',') for row in file
        ]],
        dtype=np.float32
    )
    file.close()
    blocks = int(len(X_) / n_steps)

    X_ = np.array(np.split(X_, blocks))

    return X_


# Load the networks outputs

def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # for 0-based indexing
    return y_ - 1


X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
# print X_test

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

# Input Data

training_data_count = len(X_train)
test_data_count = len(X_test)
n_input = len(X_train[0][0])

n_hidden = 32 # Hidden layer num of features
n_classes = 7



def one_hot(y_):

    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


with tf.gfile.GFile("/home/yash/Desktop/model.pb", "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

tf.import_graph_def(
    restored_graph_def,
    input_map=None,
    return_elements=None,
    name="")

graph = tf.get_default_graph()
image = graph.get_tensor_by_name("input_layer:0")
output = graph.get_tensor_by_name("output_layer:0")

with tf.Session() as sess:
    one_hot_predictions = sess.run(output, feed_dict={image: [X_test[0,:,:]]})

print(one_hot_predictions)
print(np.argmax(one_hot_predictions))
print(y_test[0])