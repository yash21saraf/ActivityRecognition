## Converting Model to pb file or tflite model. 

### 1. Conversion to .pb file. 

1. The paramaters to the meta file i.e. generated while training can be passed to the model directly here. 

The model.pb file will be saved in the Base directory where the terminal is open.
The arguments to the file are -  


```bash
- Model Name - mv2_cpm or mv2_hourglass
- Input Size - Default 192
- Checkpoint - Path to the meta file. 
- Output Node Names - default set for CPM. ("hourglass_out_3" for Hourglass model)
- Output Graph - Directory and name for the Output Graph File
```

Example - 

```bash
python -m src.converters.gen_frozen_pb --model=mv2_cpm --size=224 --checkpoint=/home/yash/ARdata/experiments/ai_challenger/trained/mv2_cpm_tiny/models/mv2_cpm_batch-16_lr-0.001_gpus-1_192x192_configurations-mv2_cpm/model-84000
```

**The input name for both the models is "image".** 

### Convert the model.pb generated to a .tflite file OR .mlmodel file. 

The .tflite file will also be saved in the Base Directory where the terminal is open. 
The arguments to the file are-

```bash
 - frozen_pb - Path of the .pb file
 - input_node_name - set as "image"
 - output_node_name - default set for CPM. ("hourglass_out_3" for Hourglass model)
 - output_path - String for output path 
 - type - default="coreml", help="tflite or coreml"
```

Example - 

```bash
python -m src.converters.gen_tflite_coreml --frozen_pb=model.pb --input_node_name=image --output_node_name=Convolutional_Pose_Machine/stage_5_out --type=tflite
```