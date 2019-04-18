import tensorflow as tf
import cv2
import numpy as np
import os
from collections import defaultdict
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_w_h = 192


frozen_graph = "/home/yash/Desktop/finalModels/pose/model.pb"
output_node_names = "Convolutional_Pose_Machine/stage_5_out"


with tf.gfile.GFile("/home/yash/Desktop/model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph1:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name="")

image1 = graph1.get_tensor_by_name("input_layer:0")
output1 = graph1.get_tensor_by_name("output_layer:0")

with tf.gfile.GFile(frozen_graph, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph2:
    tf.import_graph_def(
        graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

image2 = graph2.get_tensor_by_name("image:0")
output2 = graph2.get_tensor_by_name("%s:0" % output_node_names)



def predict_action(block, sess1):
    np.asarray(block)
    y = np.expand_dims(block, axis=0)
    one_hot_predictions = sess1.run(output1, feed_dict={image1: y})
    print(one_hot_predictions)
    print(inv_actions[np.argmax(one_hot_predictions)+1])


## Define mapping between the action and a number

actions = {"jumping in place": 1, "jumping jacks": 2, "bending(hands up all the way down)": 3,
                "punching(boxing)": 4,
                "waving(two hands)": 5, "waving(right hand)": 6, "clapping hands": 7, "throwing a ball": 8,
                # "sit and stand": 9,
                "sit down": 9, "stand up": 10}
inv_actions = {v: k for k, v in actions.items()}

if __name__ == '__main__':
    sess1 = tf.Session(graph=graph1)
    sess2 = tf.Session(graph=graph2)

    kernel = np.ones((5, 5), np.float32) / 25

    cap = cv2.VideoCapture(0)
    block = []

    while (True):
        # Capture frame-by-frame
        success, image_ = cap.read()

        if success != True:
            break

        image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

        heatmaps = sess2.run(output2, feed_dict={image2: [image_]})

        heatmaps = heatmaps[0, :, :, :]

        current_frame = []
        for i in range(14):
            heatmap_part = heatmaps[:, :, i]

            heatmap_part = cv2.filter2D(heatmap_part, -1, kernel)
            i1, j1 = np.unravel_index(heatmap_part.argmax(), heatmap_part.shape)
            current_frame.append(int(i1))
            current_frame.append(int(j1))
            image_[2 * i1, 2 * j1] = [0, 0, 255]
            image_[2 * i1 + 1, 2 * j1+1]  = [0, 0, 255]
            image_[2 * i1, 2 * j1 + 1] = [0, 0, 255]
            image_[2 * i1 +1 , 2 * j1] = [0, 0, 255]
            image_[2 * i1 -1, 2 * j1 -1] = [0, 0, 255]
            image_[2 * i1, 2 * j1 - 1] = [0, 0, 255]
            image_[2 * i1 - 1, 2 * j1] = [0, 0, 255]
            image_[2 * i1 - 1, 2 * j1 + 1] = [0, 0, 255]
            image_[2 * i1 + 1, 2 * j1 - 1] = [0, 0, 255]
        block.append(current_frame)

        frame = cv2.resize(image_, (800, 600))
        time.sleep(0.1)
        cv2.imshow('Image_Map', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(block) == 32:
            predict_action(block, sess1)
            block = block[17:32]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()