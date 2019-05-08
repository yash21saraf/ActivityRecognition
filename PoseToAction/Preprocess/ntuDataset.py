import tensorflow as tf
import cv2
import numpy as np
import os
import time
import json

####################################
from vidaug import augmentors as va

sometimes = lambda aug: va.Sometimes(0.8, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]
    va.RandomTranslate(x = 5, y = 5),
])

###################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_path = "/media/yash/YASH-01/Datasets/ABC"
input_w_h = 192
frozen_graph = "/home/yash/Desktop/finalModels/pose/model.pb"

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

kernel = np.ones((5, 5), np.float32) / 25


def read_video_and_calculate_centroids_augmentation(sess, vid_path, camera, subject, name_of_action, repetition, setting, list_of_json):
    cap = cv2.VideoCapture(vid_path)
    current_video = []
    while (cap.isOpened()):
        ret, image_ = cap.read()
        if ret == True:
            imagex = int(image_.shape[0])
            imagey = int(image_.shape[1])
            width = min(imagex, imagey * (4/3))
            height = min(imagex / (4/3), imagey)
            left = (imagex - width) / 2
            top = (imagey - height) / 2
            box = (left, top, left + width, top + height)
            image_ = image_[int(left)+15:int(left+width)-15, int(top)+20:int(top+height)-20]
            image_ = cv2.resize(image_, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)
            current_video.append(image_)
        else:
            break

    ii = 1
    while ii < 5:
        ii += 1
        video_aug = seq(current_video)
        current_vid_data = []
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
                i1 = 2 * i1
                j1 = 2 * j1
                image_[i1, j1] = [0, 0, 255]
                image_[i1 + 1, j1 + 1] = [0, 0, 255]
                image_[i1, j1 + 1] = [0, 0, 255]
                image_[i1 + 1, j1] = [0, 0, 255]
                image_[i1 - 1, j1 - 1] = [0, 0, 255]
                image_[i1, j1 - 1] = [0, 0, 255]
                image_[i1 - 1, j1] = [0, 0, 255]
                image_[i1 - 1, j1 + 1] = [0, 0, 255]
                image_[i1 + 1, j1 - 1] = [0, 0, 255]
            current_vid_data.append(current_frame_data)
            frame = cv2.resize(image_, (800, 600))
            time.sleep(0.03)
            cv2.imshow('Image_Map', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(len(current_vid_data))
        json_data = save_json(vid_path, camera, subject, name_of_action, repetition, current_vid_data, setting)
        list_of_json.append(json_data)

## The results have been saved as a json file for convenience

def save_json(vid_path, camera, subject, name_of_action, repetition, current_vid_data, setting):
    data = {
        "path": vid_path,
        "action": name_of_action,
        "camera": camera,
        "subject": subject,
        "repetition": repetition,
        "current_vid_data": current_vid_data,
        "setting": setting
    }
    return data


def writeBlock(block, file1, file2, file3, file4, subject, action):
    for frame_data in block:
        coords = ""
        for k in range(len(frame_data)):
            coords = coords + str(frame_data[k]) + ","
        coords = coords[:-1]
        if subject % 4 != 0:
            if subject % 5 != 0:
                file1.write(coords)
                file1.write("\n")
            else:
                file3.write(coords)
                file3.write("\n")

    if subject % 4 != 0:
        if subject % 5 != 0:
            file2.write(action)
            file2.write("\n")
        else:
            file4.write(action)
            file4.write("\n")



def jsonToText(frames_per_block, overlap, skip_frames):
    import json
    from random import shuffle
    filenames = os.listdir(data_path + "Experiments")

    # filenames = [data_path + "Experiments/BerkeleyMHADCam01.json",
    #              data_path + "Experiments/BerkeleyMHADCam02.json",
    #              data_path + "Experiments/BerkeleyMHADCam03.json",
    #              data_path + "Experiments/BerkeleyMHADCam04.json"]

    file1 = open(data_path + "Experiments/X_Train.txt", "w")
    file2 = open(data_path + "Experiments/Y_Train.txt", "w")
    file3 = open(data_path + "Experiments/X_Test.txt", "w")
    file4 = open(data_path + "Experiments/Y_Test.txt", "w")
    list_of_json = []

    for num, filename in enumerate(filenames):
        if filename[-4:] == "json":
            file_data = json.load(open(data_path + "Experiments/" + filename, 'r'))
            if num == 0:
                list_of_json = file_data
            else:
                list_of_json = list_of_json + file_data

    shuffle(list_of_json)
    total_vids = len(list_of_json)

    i = 0
    for video in list_of_json:
        # video_data = video["frame_data"]
        video_data = video["current_vid_data"]
        # action = actions_dict[video["action"]]
        action = str(considered_actions_dict[video["action"]])
        subject = int(video["subject"])
        block = []
        j = 0
        while (j < len(video_data)):
            block.append(video_data[j])
            if len(block) == frames_per_block:
                writeBlock(block, file1, file2, file3, file4, subject, action)
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

        settings = os.listdir(data_path)
        for setting in settings:
            list_of_json = []
            if setting[-4:] == ".txt":
                continue
            sub_dir = data_path + "/" + setting + "/" + "nturgb+d_rgb"
            allVids = os.listdir(sub_dir)
            for video in allVids:
                setting = int(video[1:4])
                camera = int(video[5:8])
                subject = int(video[9:12])
                replication = int(video[13:16])
                action = int(video[17:20])
                if action in considered_actions:
                    video_path = sub_dir + "/" + video

                    read_video_and_calculate_centroids_augmentation(sess, video_path, camera, subject, action,
                                                                       replication, setting, list_of_json)

            fileNameNew = "NTU" + str(setting) + ".json"
            writePath = data_path + "Experiments/" + fileNameNew

            if not os.path.exists(data_path + "Experiments"):
                os.makedirs(data_path + "Experiments")

            with open(writePath, 'w') as outfile:
                json.dump(list_of_json, outfile)


## Define mapping between the action and a number

considered_actions = [7,24,27,38]

considered_actions_dict = {7 : 1, 24 : 2, 27: 3, 38: 4}

f = open(data_path + "/NTUActionClass.txt", 'r')
actions_dict = {}
actions_count = {}
key = 1
for line in f:
    actions_dict[line.strip("\n")] = key
    actions_count[line.strip("\n")] = 0
    key += 1
f.close()

print(actions_dict)
inv_actions_dict = {v: k for k, v in actions_dict.items()}

if __name__ == '__main__':
    # videoToJSON(data_path)
    jsonToText(48, 40, 1)
    print(actions_count)
