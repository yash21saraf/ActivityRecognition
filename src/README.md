## Light Weight Single Person Pose Estimation

### The pose estimation code has been forked from [EdvardHua](https://github.com/edvardHua/PoseEstimationForMobile) repository. 
 
Minor bug fixes and modifications has been made. 

**Before running the code -**

1. Install the dependencies.

```bash
pip install -r requirements.txt
```

2. Beside, you also need to install [cocoapi](https://github.com/cocodataset/cocoapi)
3. Change paths in the configurations files. 
4. Also path is present in src/dataloaders.dataset.py. Update paths accordingly. 

### Training the Model - 

1. To train the model open Terminal in the src directory and run the following command. 


- For training the Convolutional Pose Machines model - 
```bash
python -m src.train.train configurations/mv2_cpm.cfg
```

- For training the Hourglass Model use - 

```bash
python -m src.train.train configurations/mv2_hourglass.cfg
```

The training parameters can be set in the [Configurations](https://github.com/yash21saraf/ActivityRecognition/tree/master/configurations) file. 
By default for the CPM model the scale has been set to 1 and the image size as 192*192. 

**Make sure you set the directory paths in the configurations folder.**

Do not change the structure of the data. Copy the data to the Home folder and then also change the username in the paths in the configurations file. 
