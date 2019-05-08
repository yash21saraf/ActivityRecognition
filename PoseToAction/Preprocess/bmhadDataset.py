import tensorflow as tf
import cv2
import numpy as np
import os
from collections import defaultdict
import time
import json


### Video Augmentation using rotation and translation
####################################
from vidaug import augmentors as va

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

###################################################################

### Define the path of the Cluster for BerkeleyMHAD Dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_path = "/media/yash/YASH-01/Datasets/BerkeleyMHAD/Camera/Cluster01"

#### Define all the Pose Estimation model details
input_w_h = 192
frozen_graph = "/home/yash/Desktop/finalModels/pose/model.pb"
output_node_names = "Convolutional_Pose_Machine/stage_5_out"
### For Hourglass Model
# output_node_names = "hourglass_out_3"

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

kernel = np.ones((5, 5), np.float32) / 25


### The nan helper is used while interpolating Nan Values
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

### The Nan interpolater averages the start and end and interpolates all the values in between
### Used while calculating moments for calculating centroid
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

### Used to read the entire dataset, use necessary augmentation and save the poses as and json file
### Input : Pose Estimation session, Video file path, Camera Number, Subject Number, Name of Action, Repetition Number, and List of JSON
def read_video_and_calculate_centroids_augmentation(sess, vid_path, camera, subject, name_of_action, repetition, list_of_json):
    image_frames_array = []
    current_vid_data = []
    frame_path = ""
    video_frames = os.listdir(vid_path)
    video_frames.sort()

    ### Add the channels to BW image and resize the same and save as an numpy array for Video Augmentation
    for frame in video_frames:
        if frame[-4:] == ".pgm":
            frame_path = vid_path + "/" + frame
            image_ = cv2.imread(frame_path, -1)
            if image_ is not None:
                image_ = np.stack((image_,) * 3, axis=-1)
                image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
                image_frames_array.append(image_)

    ### Apply transformation based on definition
    video_aug = seq(image_frames_array)

    ### Run pose estimation
    for image_ in video_aug:
        heatmaps = sess.run(output, feed_dict={image: [image_]})
        heatmaps = heatmaps[0, :, :, :]

        current_frame_data = []
        for i in range(14):
            heatmap_part = heatmaps[:, :, i]

            heatmap_part = cv2.filter2D(heatmap_part, -1, kernel)
            i1, j1 = np.unravel_index(heatmap_part.argmax(), heatmap_part.shape)
            current_frame_data.append(int(i1))
            current_frame_data.append(int(j1))
            i1 = 2*i1
            j1 = 2*j1
            image_[i1, j1] = [0, 0, 255]
            image_[i1 + 1, j1 + 1] = [0, 0, 255]
            image_[i1, j1 + 1] = [0, 0, 255]
            image_[i1 + 1, j1] = [0, 0, 255]
            image_[i1 - 1,  j1 - 1] = [0, 0, 255]
            image_[i1, j1 - 1] = [0, 0, 255]
            image_[i1 - 1, j1] = [0, 0, 255]
            image_[i1 - 1,  j1 + 1] = [0, 0, 255]
            image_[i1 + 1,  j1 - 1] = [0, 0, 255]

        frame = cv2.resize(image_, (800, 600))
        time.sleep(1)
        cv2.imshow('Image_Map', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_vid_data.append(current_frame_data)
    print(frame_path)
    print(len(current_vid_data))
    json_data = save_json(vid_path, camera, subject, name_of_action, repetition, current_vid_data)
    list_of_json.append(json_data)


## Here gaussian kernal of 5*5 size has been used and then the centroids hae been
## calculated by taking the maximas
def read_video_and_calculate_centroids(sess, vid_path, camera, subject, name_of_action, repetition, list_of_json):
    current_vid_data = []
    frame_path = ""
    video_frames = os.listdir(vid_path)
    for frame in video_frames:
        if frame[-4:] == ".pgm":
            frame_path = vid_path + "/" + frame
            image_ = cv2.imread(frame_path, -1)
            if image_ is not None:
                image_ = np.stack((image_,) * 3, axis=-1)
                image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)


                heatmaps = sess.run(output, feed_dict={image: [image_]})
                heatmaps = heatmaps[0, :, :, :]

                current_frame_data = []
                for i in range(14):
                    heatmap_part = heatmaps[:, :, i]

                    heatmap_part = cv2.filter2D(heatmap_part, -1, kernel)
                    i1, j1 = np.unravel_index(heatmap_part.argmax(), heatmap_part.shape)
                    current_frame_data.append(int(i1))
                    current_frame_data.append(int(j1))

                    image_[2 * i1, 2 * j1] = [0, 0, 255]
                    image_[2 * i1 + 1, 2 * j1 + 1] = [0, 0, 255]
                    image_[2 * i1, 2 * j1 + 1] = [0, 0, 255]
                    image_[2 * i1 + 1, 2 * j1] = [0, 0, 255]
                    image_[2 * i1 - 1, 2 * j1 - 1] = [0, 0, 255]
                    image_[2 * i1, 2 * j1 - 1] = [0, 0, 255]
                    image_[2 * i1 - 1, 2 * j1] = [0, 0, 255]
                    image_[2 * i1 - 1, 2 * j1 + 1] = [0, 0, 255]
                    image_[2 * i1 + 1, 2 * j1 - 1] = [0, 0, 255]

                frame = cv2.resize(image_, (800, 600))
                time.sleep(3)
                cv2.imshow('Image_Map', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                current_vid_data.append(current_frame_data)
    print(frame_path)
    print(len(current_vid_data))
    json_data = save_json(vid_path, camera, subject, name_of_action, repetition, current_vid_data)
    list_of_json.append(json_data)

def read_video_and_calculate_centroids_moments(sess, vid_path, camera, subject, name_of_action, repetition, list_of_json):
    x = defaultdict(list)
    y = defaultdict(list)
    frame_path = ""
    video_frames = os.listdir(vid_path)
    for frame in video_frames:
        if frame[-4:] == ".pgm":
            frame_path = vid_path + "/" + frame
            image_ = cv2.imread(frame_path, -1)
            if image_ is not None:
                image_ = np.stack((image_,) * 3, axis=-1)
                image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
                heatmaps = sess.run(output, feed_dict={image: [image_]})
                heatmaps = heatmaps[0, :, :, :]

                current_frame_data = []
                for i in range(14):
                    heatmap_part = heatmaps[:, :, i]

                    heatmap_part = cv2.filter2D(heatmap_part, -1, kernel)

                    ret, thresh = cv2.threshold(heatmap_part, 0.3, 1, 0)
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
    print(frame_path)
    json_data = save_json_moments(vid_path, camera, subject, name_of_action, repetition, x, y)
    list_of_json.append(json_data)

def save_json_moments(vid_path, camera, subject, name_of_action, repetition, xs, ys):
    frames = len(xs[0])
    frame_data = []
    for i in range(frames):
        coordinates = []
        for j in range(14):
            coordinates.append(xs[j][i])
            coordinates.append(ys[j][i])
        frame_data.append(coordinates)
    data = {
        "path": vid_path,
        "action": name_of_action,
        "camera": camera,
        "subject": subject,
        "repetition": repetition,
        "current_vid_data": frame_data
    }
    return data

## The results have been saved as a json file for convenience

def save_json(vid_path, camera, subject, name_of_action, repetition, current_vid_data):
    data = {
        "path": vid_path,
        "action": name_of_action,
        "camera": camera,
        "subject": subject,
        "repetition": repetition,
        "current_vid_data": current_vid_data
    }
    return data

counter = 0

def writeBlock(block, file1, file2, file3, file4, action):
    global counter
    for frame_data in block:
        coords = ""
        for k in range(len(frame_data)):
            coords = coords + str(frame_data[k]) + ","
        coords = coords[:-1]
        if counter < 4:
            file1.write(coords)
            file1.write("\n")
        else:
            file3.write(coords)
            file3.write("\n")

    if counter < 4:
        file2.write(action)
        file2.write("\n")
        counter += 1
    else:
        file4.write(action)
        file4.write("\n")
        counter = 0

def jsonToText(frames_per_block, overlap, skip_frames):
    import json
    from random import shuffle

    filenames = [data_path + "Experiments/BerkeleyMHADCam01.json",
                 data_path + "Experiments/BerkeleyMHADCam02.json",
                 data_path + "Experiments/BerkeleyMHADCam03.json",
                 data_path + "Experiments/BerkeleyMHADCam04.json"]

    file1 = open(data_path + "Experiments/X_Train.txt", "w")
    file2 = open(data_path + "Experiments/Y_Train.txt", "w")
    file3 = open(data_path + "Experiments/X_Test.txt", "w")
    file4 = open(data_path + "Experiments/Y_Test.txt", "w")
    list_of_json = []

    for num, filename in enumerate(filenames):
        file_data = json.load(open(filename, 'r'))
        if num == 0:
            list_of_json = file_data
        else:
            list_of_json = list_of_json + file_data

    shuffle(list_of_json)

    i = 0
    for video in list_of_json:
        # video_data = video["frame_data"]
        video_data = video["current_vid_data"]
        if video["action"] == "sit and stand":
            continue
        action = str(actions_dict[video["action"]])
        block = []
        j = 0
        while (j < len(video_data)):
            block.append(video_data[j])
            if len(block) == frames_per_block:
                if actions_count[video["action"]] >= 884:
                    continue
                else:
                    actions_count[video["action"]] += 1
                writeBlock(block, file1, file2, file3, file4, action)
                block.clear()
                j -= overlap
            j += skip_frames
        i += 1

    file1.close()
    file2.close()
    file3.close()
    file4.close()

def videoToJSON(data_path):
    with tf.Session() as sess:
        ## Traversing into the directory structure and passing the file to the model by calling function
        cameras = os.listdir(data_path)
        for camera in cameras:
            list_of_json = []
            subjects = os.listdir(data_path + "/" + camera)
            for subject in subjects:
                actions = os.listdir(data_path + "/" + camera + "/" + subject)
                for action in actions:
                    repetitions = os.listdir(data_path + "/" + camera + "/" + subject + "/" + action)
                    for repetition in repetitions:
                        if action == "T-pose":
                            name_of_action = "T pose"
                        else:
                            name_of_action = inv_actions_dict[int(str(action)[1:])]
                        if action[0] != "B" and name_of_action != "T pose":
                            if name_of_action == "clapping hands":
                                    vid_path = data_path + "/" + camera + "/" + subject + "/" + action + "/" + repetition
                                    read_video_and_calculate_centroids_augmentation(sess, vid_path, camera, subject, name_of_action,
                                                                       repetition, list_of_json)

            fileNameNew = "BerkeleyMHAD" + camera + ".json"
            writePath = data_path + "Experiments/" + fileNameNew

            if not os.path.exists(data_path + "Experiments"):
                os.makedirs(data_path + "Experiments")

            with open(writePath, 'w') as outfile:
                json.dump(list_of_json, outfile)


## Define mapping between the action and a number

actions_dict = {"jumping in place": 1, "jumping jacks": 2, "bending(hands up all the way down)": 3,
                "punching(boxing)": 4,
                "waving(two hands)": 5, "waving(right hand)": 6, "clapping hands": 7, "throwing a ball": 8,
                "sit down": 9, "stand up": 10}


actions_count = {"jumping in place": 0, "jumping jacks": 0, "bending(hands up all the way down)": 0,
                 "punching(boxing)": 0, "waving(two hands)": 0, "waving(right hand)": 0, "clapping hands": 0,
                 "throwing a ball": 0, "sit down": 0, "stand up": 0}

inv_actions_dict = {v: k for k, v in actions_dict.items()}

if __name__ == '__main__':
    videoToJSON(data_path)
    # jsonToText(10, 8, 4)
    print(actions_count)
