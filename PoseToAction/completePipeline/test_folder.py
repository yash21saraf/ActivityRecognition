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

### Update paths to model files

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
    print(inv_actions[considered_actions[np.argmax(one_hot_predictions)]])


## Define mapping between the action and a number


data_path = "/media/yash/Data/Dataset/Testing"
action_index_path = "/media/yash/YASH-01/Datasets/ABC"
f = open(action_index_path + "/NTUActionClass.txt", 'r')
actions = {}
actions_count = {}
key = 1
for line in f:
    actions[line.strip("\n")] = key
    actions_count[line.strip("\n")] = 0
    key += 1
f.close()

print(actions)
inv_actions = {v: k for k, v in actions.items()}

considered_actions = [7,24,27,38]

considered_actions_dict = {7 : 1, 24 : 2, 27: 3, 38: 4}

if __name__ == '__main__':
    sess1 = tf.Session(graph=graph1)
    sess2 = tf.Session(graph=graph2)

    kernel = np.ones((5, 5), np.float32) / 25

    allVids = os.listdir(data_path)
    for video in allVids:
        block = []
        video_path = data_path + "/" + video
        cap = cv2.VideoCapture(video_path)

        while (cap.isOpened()):
            ret, image_ = cap.read()
            if ret == True:

                imagex = int(image_.shape[0])
                imagey = int(image_.shape[1])
                width = min(imagex, imagey * (4 / 3))
                height = min(imagex / (4 / 3), imagey)
                left = (imagex - width) / 2
                top = (imagey - height) / 2
                box = (left, top, left + width, top + height)
                # image_ = image_[int(left) + 15:int(left + width) - 15, int(top) + 20:int(top + height) - 20]
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

                if len(block) == 48:
                    predict_action(block, sess1)
                    block = block[17:48]
            else:
                break
            # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

