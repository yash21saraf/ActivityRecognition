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

with tf.gfile.GFile(frozen_graph, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())

tf.import_graph_def(
    restored_graph_def,
    input_map=None,
    return_elements=None,
    name=""
)

graph = tf.get_default_graph()
image = graph.get_tensor_by_name("image:0")
output = graph.get_tensor_by_name("%s:0" % output_node_names)


def display_joint(pred_heat=None):
    cv2.imshow('HeatMap', cv2.resize(pred_heat, (800, 600)))
    cv2.waitKey(1)


def display_image(pred_heat=None):
    tmp = np.amax(pred_heat, axis=2)
    frame = cv2.resize(tmp, (800, 600))
    time.sleep(0.1)
    cv2.imshow('HeatMap', frame)
    cv2.waitKey(1)


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def nan_interpolator(centroid_dict):
    for key, value in centroid_dict.items():
        new_value = np.asarray(value)
        nans, x = nan_helper(new_value)
        if not np.isnan(new_value).all():
            new_value[nans] = np.interp(x(nans), x(~nans), new_value[~nans])
        else:
            new_value = [-1000.0] * len(value)
        new_value = np.asarray(new_value)
        new_value = new_value.astype(np.float64)
        centroid_dict[key] = new_value


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


## Here centroid of blob has been calculated using Moments
## Someplaces the values may be returned as Nan. So Nan interpolator
## has been used to fill up these values

def read_video_and_calculate_centroids(sess, file_name, vid_path, video_class, list_of_json):
    x = defaultdict(list)
    y = defaultdict(list)
    video = cv2.VideoCapture(vid_path)

    while (video.isOpened()):
        success, image_ = video.read()
        if success != True:
            break

        image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

        heatmaps = sess.run(output, feed_dict={image: [image_]})

        heatmaps = heatmaps[0, :, :, :]

        # display_joint(heatmaps[:,:,1])
        for i in range(14):
            heatmap_part = heatmaps[:, :, i]

            heatmap_part = 2 * np.array(heatmap_part)
            heatmap_part[heatmap_part > 1] = 1

            ret, thresh = cv2.threshold(heatmap_part, 0.3, 1, 0)

            # calculate moments of binary image
            M = cv2.moments(thresh)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = float('nan'), float('nan')

            x[i].append(cX)
            y[i].append(cY)
    nan_interpolator(x)
    nan_interpolator(y)
    json_data = save_json(file_name, vid_path, video_class, x, y)
    list_of_json.append(json_data)
    # visualize_centroid(x,y,2)
    video.release()


## Here gaussian kernal of 5*5 size has been used and then the centroids hae been
## calculated by taking the maximas
def read_video_and_calculate_centroids_new(sess, file_name, vid_path, video_class, list_of_json):
    x = defaultdict(list)
    y = defaultdict(list)
    video = cv2.VideoCapture(vid_path)
    kernel = np.ones((5, 5), np.float32) / 25
    while (video.isOpened()):
        success, image_ = video.read()
        if success != True:
            break

        image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

        heatmaps = sess.run(output, feed_dict={image: [image_]})

        heatmaps = heatmaps[0, :, :, :]

        # display_joint(heatmaps[:,:,1])
        for i in range(14):
            heatmap_part = heatmaps[:, :, i]

            heatmap_part = cv2.filter2D(heatmap_part, -1, kernel)
            i1, j1 = np.unravel_index(heatmap_part.argmax(), heatmap_part.shape)
            x[i].append(int(i1))
            y[i].append(int(j1))
    json_data = save_json(file_name, vid_path, video_class, x, y)
    list_of_json.append(json_data)
    # visualize_centroid(x,y,2)
    video.release()


## The results have been saved as a json file for convenience

def save_json(video_id, path, action, xs, ys):
    frames = len(xs[0])
    frame_data = []
    for i in range(frames):
        coordinates = []
        for j in range(14):
            coordinates.append(xs[j][i])
            coordinates.append(ys[j][i])
        frame_data.append(coordinates)
    data = {
        "video_id": video_id,
        "path": path,
        "action": action,
        "num_frames": frames,
        "frame_data": frame_data
    }
    return data


def writeBlock(block, file1, file2, file3, file4, i, total_vids, action):
    for frame_data in block:
        coords = ""
        for k in range(len(frame_data)):
            coords = coords + str(frame_data[k]) + ","
        coords = coords[:-1]

        if i < 0.8 * total_vids:
            file1.write(coords)
            file1.write("\n")

        else:
            file3.write(coords)
            file3.write("\n")

    if i < 0.8 * total_vids:
        file2.write(action)
        file2.write("\n")

    else:
        file4.write(action)
        file4.write("\n")


def jsonToText(frames_per_block, overlap):
    import json
    from random import shuffle

    filename = data_path + "Experiments/DronesDataMoments.json"
    file1 = open(data_path + "Experiments/X_Train_moments.txt", "w")
    file2 = open(data_path + "Experiments/Y_Train_moments.txt", "w")
    file3 = open(data_path + "Experiments/X_Test_moments.txt", "w")
    file4 = open(data_path + "Experiments/Y_Test_moments.txt", "w")

    list_of_json = json.load(open(filename, 'r'))
    shuffle(list_of_json)
    total_vids = len(list_of_json)

    i = 0
    for video in list_of_json:
        video_data = video["frame_data"]
        action = str(actions[video["action"]])
        block = []
        j = 0
        while(j < len(video_data)):
        # for j in range(len(video_data)):
            block.append(video_data[j])
            if len(block) == frames_per_block:
                writeBlock(block, file1, file2, file3, file4, i, total_vids,action)
                block.clear()
                j -= overlap
            j+=1
        i += 1

    file1.close()
    file2.close()
    file3.close()
    file4.close()


def videoToJSON(data_path):
    list_of_json = []
    with tf.Session() as sess:

        ## Traversing into the directory structure and passing the file to the model by calling function

        parent = os.listdir(data_path)
        for video_class in parent:

            child = os.listdir(data_path + "/" + video_class)
            for class_i in child:

                sub_child = os.listdir(data_path + "/" + video_class + "/" + class_i)
                for file in sub_child:
                    vid_path = data_path + "/" + video_class + "/" + class_i + "/" + file
                    read_video_and_calculate_centroids(sess, file, vid_path, video_class, list_of_json)

    writePath = data_path + "Experiments/DronesDataMoments.json"

    if not os.path.exists(data_path + "Experiments"):
        os.makedirs(data_path + "Experiments")

    with open(writePath, 'w') as outfile:
        json.dump(list_of_json, outfile)


## Define mapping between the action and a number

actions = {"box": 1, "clap": 2, "Jog": 3, "sit": 4, "stand": 5, "walk": 6, "wave": 7}

if __name__ == '__main__':
    videoToJSON(data_path)
    jsonToText(48,24)
