# Activity_Recognition

The problem of Activity Recognition is being studied using Recurrent Neural Networks on Human Pose Estimates. 

For, task simplicity we started with Single Person Pose Estimatin, where we have data collected by DRONES lab at UB for 7 classes. For the calculation of pose estimates the following Github Repository has been referred. 

[Page not found · GitHub · GitHub](https://github.com/edvardHua/PoseEstimationForMobile)

This is the code by Edvard Hua for application of Pose Estimation on Android Devices using tflite and MACE. 

The model training, model, tflite conversion code along with the Android Code is included in the repository. So, the pose extracted using the Repository returns 14 keypoints. 

So, as a starting approach the LSTM network would be fed with the 14 * 2 points for each frame. Using a LSTM network the action would be predicted. 

The WorkingDir has been created in the src folder to work on the same problem. 
The test1.py is used to read the dataset and store the frame wise poses as an JSON format. 

The Pose Estimation model returns an Array of dimensions [1, width, height, 14], where all 14 arrays contains a HeatMap for a particular Joint.

So, using Blob Detection the centroids of the HeatMaps are generated. Since, Blob Detection returns null values for some frames, interpolation has been used to fill in the remaining values. 

Right now, working on the task of reading the JSON data and feeding it to an LSTM network to obtain a baseline model. 

## The link for the dataset and Models: 

https://drive.google.com/drive/folders/1iyhklD5GUgLaT5MNza6jYRIyxjZnkQQ4?usp=sharing

The AI challeneger dataset can be found in hdd along with the checkpoints and tensorflow graph files. 
The DRONES lab dataset is present as RadhaDataset which contains 7 classes. 

The Pose Estimation has been trained on AI challenger. Using the model the poses are extracted from the RadhaDataset and saved as data.json in the training folder. The poses will now be used to train the LSTM network. 

The Document links can be found below - 

- Requirements Document - 
https://docs.google.com/document/d/1l4sXT5vA_gSxC-7S-urvFv1tsdUcsJjET5-kk704RMo/edit?usp=sharing

- Technical Design - 
https://docs.google.com/document/d/1V-XI6nPnCq88g-U6pkyce6Mv6UvVp5hW-nDUCUkS1w4/edit

- Project Plan 
https://docs.google.com/spreadsheets/d/1Yc387e71iDSWl4gIgV49CVoIBxLj38UCaLazNXwxMSI/edit?usp=sharing

- Project Updates
https://docs.google.com/document/d/1nL22u8UdTu127q7C94zHcQsjrAymY89q2r7TNtycJfg/edit?usp=sharing
# ActivityRecognition
