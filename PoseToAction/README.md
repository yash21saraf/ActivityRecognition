## Action Recognition 

For action recognition the dataset being used was created at DRONES lab. The details of the dataset can be seen here. 
[Dataset Details]().

For the basic approach this [Github Repository](https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input) is being followed for LSTM model architecture. 

For dataset preprocessing check Preprocess. Here JSON data created using the Pose Estmation has been 
converted to a txt file. 

### Preprocessing

For details related to preprocessing check here [Preprocess](Preprocess).

- **The output of the Preprocessing will be 4 text files i.e. X_Train, X_Test, Y_Train, Y_Test. These 
files are the input for the LSTM model.**

- 1 important parameter to set before training is the num_steps which depends upon the block
size used for Preprocessing. 
- For the results shared below the blocksize of 48 is used along with the overlap of 32. 


Run the following command to from Base folder to generate the JSON, and text files corresponding to the video files.

```bash
python -m PoseToAction.Preprocess.dronesDataset
```


### Training 

For training the model set the path in the newtworks.network.py and then run the following command.

```bash
python -m PoseToAction.networks.network
```
