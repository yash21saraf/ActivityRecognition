### TRAINING 

The model is a simple 2 cell LSTM network which takes a block of Pose Values as an input, 
The input dimensions depend upon the duration of timestep which the LSTM needs to see to predict a 
particular action. 

So, for different datasets, different values have been experimented based on frame rates.

The following are the parameters which have been used with the NTU Dataset. 

```bash

Input Dimensions: [48, 28] // 48 frame data, 14*2 coordinates in 2D plane(keypoints)
Output Dimensions: [,4] // Probability for each class
Hidden Params: 32 per cell 
Number of LSTM cells: 2 

```

Using the above vanilla LSTM model, following accuracy has been obtained - 83 percent for 4 class classification 

Following is the representation for the Confusion Matrix- 
![image](https://github.com/yash21saraf/ActivityRecognition/blob/master/images/NTUConfusion.png)

Following is the training curve - 

![image](https://github.com/yash21saraf/ActivityRecognition/blob/master/images/NTUTraining.png)

In total 40 subjects were part of the dataset creation. So to make sure the testing has been done on completely unseen dataset,
10 subjects were seperated out. For training and validation rest of the dataset was used. 

This made sure that there was no subject bias involved during testing the model. 

### Checkpoint To PB and tflite Converter

The checkpoint to PB file as the name suggests is used to create a PB file which can be later used for
converting the model to a tflite file. 

The purpose of the project is to try the model out on edge device. For our study we have used an android device as
an edge device and trying the model on the same. 

The same params i.e. input layer name, output layer name, number of frames, number of classes, and path to the dataset has 
been defined in all the files. Make necessary changes according to the dataset being used. 



