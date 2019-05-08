### ACTION RECOGNITION 

Here all the files take in 2 model files i.e. the Pose Estimation model which is being used to calculate the 14 keypoints, and the 
action recognition LSTM model which takes data from consecutive frames and predicts the action. 

So, the path to thee model file needs to be defined in each file. For the NTU dataset the classes are being 
read from the txt file. So the path to the txt file also needs to be defined. 

The test_ntu takes the subject left out during training and runs the Model. The results can be seen in the following video. 
The path t0 the dataset needs to be defined. 

Whereas the test_folder runs the model on all the video files present in the folder. The test camera 
takes the input of the webcam and runs the model live. 

[![Model Performance](https://youtu.be/tCYAVXMYee0/0.jpg)](https://youtu.be/tCYAVXMYee0)
