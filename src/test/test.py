# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-


def display_image():
    """
    display heatmap & origin image
    :return:
    """

    from src.dataloaders.dataset_prepare import CocoMetadata, CocoPose
    from pycocotools.coco import COCO
    from os.path import join
    from src.dataloaders.dataset import _parse_function

    BASE_PATH = "/home/yash/ARdata/ai_challenger"

    ANNO = COCO(
        join(BASE_PATH, "ai_challenger_valid.json")
    )
    train_imgIds = ANNO.getImgIds()

    img, heat = _parse_function(train_imgIds[100],False ,ANNO)

    CocoPose.display_image(img, heat, pred_heat=heat, as_numpy=False)

    ## The CocoPose class's display image returns the data as an numpy array which can be used to save the file as
    ## an image.
    ## To understand the conversion open dataloaders.dataset_prepare where the method has been defined.

    from PIL import Image
    import numpy as np
    for _ in range(heat.shape[2]):
        data = CocoPose.display_image(img, heat, pred_heat=heat[:, :, _:(_ + 1)], as_numpy=True)
        im = Image.fromarray(data)



## Imports the network architecture from networks and saves the model log file to
## tensorboard/test_graph directory for CPM model.

def saved_model_graph_cpm():
    """
    save the graph of model and check it in tensorboard
    :return:
    """

    from os.path import join
    from src.networks.network_mv2_cpm import build_network
    import tensorflow as tf
    import os

    INPUT_WIDTH = 224
    INPUT_HEIGHT = 224
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_HEIGHT, 3),
                                name='image')
    build_network(input_node, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(
            join("tensorboard/test_graph/"),
            sess.graph
        )
        sess.run(tf.global_variables_initializer())

## Imports the network architecture from networks and saves the model log file to
## tensorboard/test_graph directory for Hourglass model.

def saved_model_graph_hourglass():
    """
    save the graph of model and check it in tensorboard
    :return:
    """

    from os.path import join
    from src.networks.network_mv2_hourglass import build_network
    import tensorflow as tf
    import os

    INPUT_WIDTH = 224
    INPUT_HEIGHT = 224
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_node = tf.placeholder(tf.float32, shape=(1, INPUT_WIDTH, INPUT_HEIGHT, 3),
                                name='image')
    build_network(input_node, False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(
            join("tensorboard/test_graph/"),
            sess.graph
        )
        sess.run(tf.global_variables_initializer())

## Used for profiling the model
## TODO: Check Documentation

def metric_prefix(input_width, input_height):
    """
    output the calculation of you model
    :param input_width:
    :param input_height:
    :return:
    """
    import tensorflow as tf
    from src.networks.networks import get_network
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    input_node = tf.placeholder(tf.float32, shape=(1, input_width, input_height, 3),
                                name='image')
    get_network("mv2_cpm", input_node, False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_meta = tf.RunMetadata()
    with tf.Session(config=config) as sess:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("opts {:,} --- paras {:,}".format(flops.total_float_ops, params.total_parameters))
        sess.run(tf.global_variables_initializer())

## Picks up the model.pb from directory and runs the model on a given image
def run_with_frozen_pb(img_path, input_w_h, frozen_graph, output_node_names):
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from src.dataloaders.dataset_prepare import CocoPose
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name="")

    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image:0")
    output = graph.get_tensor_by_name("%s:0" % output_node_names)

    image_0 = cv2.imread(img_path)
    w, h, _ = image_0.shape
    image_ = cv2.resize(image_0, (input_w_h, input_w_h), interpolation=cv2.INTER_AREA)

    with tf.Session() as sess:
        heatmaps = sess.run(output, feed_dict={image: [image_]})

        CocoPose.display_image(
            image_,
            None,
            heatmaps[0,:,:,:],
            False
        )


        # save each heatmaps to disk
        from PIL import Image
        heatmaps = heatmaps[0,:,:,:]
        for _ in range(heatmaps.shape[2]):
            data = CocoPose.display_image(image_, heatmaps, pred_heat=heatmaps[:, :, _:(_ + 1)], as_numpy=True)
            im = Image.fromarray(data)
            # im.save("/home/yash/ARdata//PoseEstimationForMobile/training/heat_%d.jpg" % _)


if __name__ == '__main__':

    # saved_model_graph_cpm()
    # metric_prefix(224, 224)
    run_with_frozen_pb(
        "/home/yash/ARdata/ai_challenger/train/0a9396675bf14580eb08c37e0b8a69a0299afb03.jpg",
        224,
        "/home/yash/Desktop/Human_Pose/PoseEstimationForMobile/training/yash.pb",
        "Convolutional_Pose_Machine/stage_5_out"
    )
    #display_image()