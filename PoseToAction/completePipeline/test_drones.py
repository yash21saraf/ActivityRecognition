import tensorflow as tf
import cv2
import numpy as np
import os
from collections import defaultdict
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data_path = "/home/yash/ARdata/DRONESDataset"
input_w_h = 224


frozen_graph = "/home/yash/Desktop/Human_Pose/PoseEstimationForMobile/training/yash.pb"
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


def display_joint(pred_heat=None):
    cv2.imshow('HeatMap', cv2.resize(pred_heat, (800, 600)))
    cv2.waitKey(1)

def display_image(pred_heat=None):
    tmp = np.amax(pred_heat, axis=2)
    frame = cv2.resize(tmp, (800, 600))
    time.sleep(0.1)
    cv2.imshow('HeatMap', frame)
    cv2.waitKey(1)

def visualize_centroid(d1, d2, part):
    arr = []
    img = np.zeros((112, 112, 3), np.uint8)
    xs = d1[part]
    ys = d2[part]
    for i in range(len(xs)):
        img = cv2.circle(img, (int(xs[i]), int(ys[i])), 1, (0, 0, 255), -1)
        # img = cv2.resize(img, (800, 600))
        time.sleep(0.1)
        cv2.imshow('ImageWindow', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def read_video_and_calculate_centroids_new(sess1, sess2, file_name, vid_path, video_class):
    block = []
    video = cv2.VideoCapture(vid_path)
    kernel = np.ones((5, 5), np.float32) / 25
    while (video.isOpened()):
        success, image_ = video.read()
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
            image_[2*i1,2*j1] = [0,0,255]
        block.append(current_frame)

        frame = cv2.resize(image_, (800, 600))
        time.sleep(0.1)
        cv2.imshow('Image_Map', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(block) == 48:
            predict_action(block, sess1)
            block = block[17:48]
    video.release()



def predict_action(block, sess1):
    np.asarray(block)
    y = np.expand_dims(block, axis=0)
    one_hot_predictions = sess1.run(output1, feed_dict={image1: y})
    print(inv_actions[np.argmax(one_hot_predictions)+1])

def videoToJSON(data_path):

    sess1 = tf.Session(graph=graph1)
    sess2 = tf.Session(graph=graph2)
    ## Traversing into the directory structure and passing the file to the model by calling function
    count = 0
    parent = os.listdir(data_path)
    for video_class in parent:
        if count == 1:
            break
        child = os.listdir(data_path + "/" + video_class)
        for class_i in child:
            if count == 1:
                break

            sub_child = os.listdir(data_path + "/" + video_class + "/" + class_i)
            for file in sub_child:
                if count == 1:
                    break
                vid_path = data_path + "/" + video_class + "/" + class_i + "/" + file
                print(vid_path)
                read_video_and_calculate_centroids_new(sess1, sess2, file, vid_path, video_class)
                # count += 1


## Define mapping between the action and a number

actions = {"box": 1, "clap": 2, "Jog": 3, "sit": 4, "stand": 5, "walk": 6, "wave": 7}
inv_actions = {v: k for k, v in actions.items()}

if __name__ == '__main__':
    videoToJSON(data_path)
