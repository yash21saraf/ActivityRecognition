## AI Challeneger Dataset. 


### Dataset:

Training dataset.

Unzip it will obtain the following file structure

```bash
$ tree -L 1 .
.
├── ai_challenger_train.json
├── ai_challenger_valid.json
├── train
└── valid
```

The training dataset only contains single person images and it come from the competition of [AI Challenger](https://challenger.ai/datasets/keypoint). 

* 22446 training examples
* 1500 testing examples

The data annotations has been transferred to the COCO format for using the data augment code from [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) respository by the author Edvard Hua. 
