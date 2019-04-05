# Activity_Recognition

Activity Recognition has been studied using Human Pose Estimation. The details of the code have and how to run them have been
added to the respective README.

The first approach has been done using the Light Weight Pose Estimation by Edvard Hua. Details
for the pose estimation can be found [here](https://github.com/yash21saraf/ActivityRecognition/tree/master/src).

The poses extracted using Pose Estimation have been fed to the LSTM network with 2 cells. 

The details and code is available in PoseToAction Directory. 

These are the results of the model as of now. 


### Results

This is the confusion matrix for the model. 

It can be clearly seen that the model is unable to differentiate between similar actions.
For example the model confuses jogging and walking, boxing and clapping as the motion and poses are quite close to each other. 

![image](https://github.com/yash21saraf/ActivityRecognition/blob/master/images/confusionAction.png)


The model accuracy of different paramter setting for block size and overlap lies between the range of 45-55 percent. 

This is decent for a base model. The dataset used here is very noisy. No preprocessing has been
done here to account for camera movements, occlusions, missing parts, Low light, subject size, relative position to frame, etc

Addressing the above issues will help finetune the model. 


### Current Task

- Retraining the pose estimastion model with the scale value 1. 
- Now the model only returns heatmaps for the pose. 
- Outputting the features extracted by the mobilenet model along with the heatmaps and feeding 
then to the LSTM. 
- Trying other datasets like [UC Berkeley MHAD Dataset](http://tele-immersion.citris-uc.org/berkeley_mhad).
- Stitching models for mobile Device implementation. 



### [Dataset Drive Link](https://drive.google.com/drive/folders/1m0StuUeys0jz8hAaxmykEgHIS7EpIIgv?usp=sharing)