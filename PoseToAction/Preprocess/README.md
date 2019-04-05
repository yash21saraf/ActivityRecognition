### DRONES Lab Action Recognition Dataset 

**Dataset has been created by Radhakrishna Dasari while working with
Drones Lab while working on the task of Vision Learning.**

The directory structure for the dataset is following-

```bash
(base) yash@yash:~/ARdata/RadhaDataset$ tree -L 2
.
├── box
│   ├── normal
│   ├── rotandtrans
│   └── rotation
├── clap
│   ├── normal
│   ├── rotandtrans
│   └── rotation
├── Jog
│   ├── normal
│   ├── rotandtrans
│   └── rotation
├── sit
│   ├── normal
│   ├── rotandtrans
│   └── rotation
├── stand
│   ├── normal
│   ├── rotandtrans
│   └── rotation
├── walk
│   ├── normal
│   ├── rotandtrans
│   └── rotation
└── wave
    ├── normal
    ├── rotandtrans
    └── rotation

```

Here data augmentation has already been done. The Dataset consists of the above mentioned 7 classes. 
So, to read these video files the dronesDataset.py code traverses the entire directory structure uses model.pb created 
using Pose Estimation to find 14 keypoints and saves the output as a JSON format with the following structure. 


This is the structure for the JSON object created. On running pose estimation we get 14 keypoints i.e. 
(x,y) pairs which have been concatenated in the following manner - 

> (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14)
 
where 1-14 numbers represent a particular join as mentioned below - 

```bash
    Top = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    Background = 14
   ```
   
The JSON object created now can serve as an input to the Action Classifier model. 

### Dataset Creation - 

1. The JSON file is first read and shuffled. In the dataset we have a total of 427 videos. 
2. After shuffling the data has been converted to following format. 

In the input data we have X,Y where X represents a 28 value vector of Pose keypoints as represented above. 
The Y values are 1-7 depending upon the class. Following is the mapping for the same. 

```javascript
actions = {"box": 1, 
           "clap" : 2, 
           "Jog" : 3, 
           "sit" : 4, 
           "stand" : 5, 
           "walk" : 6 , 
           "wave" : 7}
```

### Code Parameters

- Modify the path in the dronesDataset.py file accordingly. 
- Add the path for the model.pb file. (For creating model.pb file refer [Pose Estimation]())
- Based on the architecture used add the name of the final layer of the model. 
    - For CPM model - Convolutional_Pose_Machine/stage_5_out
    - For Hourglass Model - hourglass_out_3
- Also add the input dimensions used while training the model.pb. 
- Also the data has been seperated into blocks. For the above dataset the video frame rate is 30 fps.
As a reasonable assumption data has been taken for 1.5 seconds i.e. 48 frames. And to augment the dataset size
the overlap between two blocks have been set to 32. These parameters can be passed to the function.
- **The output of the file will be 4 text files i.e. X_Train, X_Test, Y_Train, Y_Test. These 
files are the input for the LSTM model.**
### Implementation details

- The model.pb is used for extracting the poses from the image. 
- The model.pb returns a 4 dimensional tensor [0,image_width,image_height,14]. 
This tensor basically has 14 arrays each containing one keypoint. 
- The output heatmap dimensions are dependent on the scale set while training the model.
- Heatmap dimension = Input image/scale
- So, first method used was complicated which involved blob detection and interpolation of missing frames. 
- Looking at the implementation details in the [PoseEstimation]() application the 
implementation has been modified. Now the gaussian filter of 5*5 has been applied to
all 14 heatmaps and then index of the maimum value is used as the keypoint. 
- Visualization functions were also created to verify the output of the model on the 
video stream. 

**Use the following command from project Directory once all the parameters have been set-** 

```bash
python -m PoseToAction.Preprocess.dronesDataset
```

