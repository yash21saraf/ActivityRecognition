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

### MS Project Requirement Documents - 

The Document links can be found below - 

- [Requirements Document](https://docs.google.com/document/d/1l4sXT5vA_gSxC-7S-urvFv1tsdUcsJjET5-kk704RMo/edit?usp=sharing)

- [Technical Design](https://docs.google.com/document/d/1V-XI6nPnCq88g-U6pkyce6Mv6UvVp5hW-nDUCUkS1w4/edit)

- [Project Plan](https://docs.google.com/spreadsheets/d/1Yc387e71iDSWl4gIgV49CVoIBxLj38UCaLazNXwxMSI/edit?usp=sharing)

- [Project Updates](https://docs.google.com/document/d/1nL22u8UdTu127q7C94zHcQsjrAymY89q2r7TNtycJfg/edit?usp=sharing)

- [Test Plan](https://docs.google.com/document/d/1hLGOwoi4ub9VKGZYFguZkNeVEmiGH68hpVgesnLx-Xs/edit)


