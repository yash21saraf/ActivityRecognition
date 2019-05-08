# Activity_Recognition


The code will be updated very soon, the code requires better abstraction for reading data and processing.

Activity Recognition has been studied using Human Pose Estimation. The details of the code have and how to run them have been
added to the respective README.

The first approach has been done using the Light Weight Pose Estimation by Edvard Hua. Details
for the pose estimation can be found [here](https://github.com/yash21saraf/ActivityRecognition/tree/master/src).

The poses extracted using Pose Estimation have been fed to the LSTM network with 2 cells. 

The details and code is available in PoseToAction Directory. 

These are the results of the model as of now. 


### Results

After working with Berkeley MHAD and DRONES lab dataset, we realized that the pose estimation being used
for the models was not very accurate and needed to be tested on a more clean dataset. 

So, NTU RGB-D Dataset provided with ample training data which has been recorded in a controlled environment in colored in 
a well lit room. 

So the Pose estimation on the data was performed, and the values were saved as the JSON file. For starters, we 
only considered 4 classes i.e. throw, kick, jump, and salute. The results for the same can be found here. 

[![Model Performance](https://youtu.be/tCYAVXMYee0/0.jpg)](https://youtu.be/tCYAVXMYee0)


Following is the representation for the Confusion Matrix- 
![image](https://github.com/yash21saraf/ActivityRecognition/blob/master/images/NTUConfusion.png)

Following is the training curve - 

![image](https://github.com/yash21saraf/ActivityRecognition/blob/master/images/NTUConfusion.png)

In total 40 subjects were part of the dataset creation. So to make sure the testing has been done on completely unseen dataset,
10 subjects were seperated out. For training and validation rest of the dataset was used. 
