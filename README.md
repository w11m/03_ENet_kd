# ENet_kd
this is an ENet knowledge distillation

## Introduction
This code works for semantic segmentation (ENet) knowledge distillation 

The master branch works with **Pytorch 1.7.0**, with **Python 3.5,3.6**.

### Training Data

This implementation has been tested on the CamVid and Cityscapes datasets.

CamVid Dataset : [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
or use the terminal command below.
```Shell
wget https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/camvid.tgz
tar -xzf camvid.tgz
```
The directory structure of this dataset is shown below. In the CamVid dataset, both training and annotation data are binary image files.
```Shell
    CamVid
     ├─ label_colors.txt
     │
     ├─ train.txt
     ├─ train
     │   ├─ 0001TP_006690.png
     │   ├─ 0001TP_007500.png
     │   └─ ...
     ├─ trainannot
     │   ├─ 0001TP_006690.png
     │   ├─ 0001TP_007500.png
     │   └─ ...
     │
     ├─ val.txt
     ├─ val
     │   ├─ 0016E5_07959.png
     │   ├─ 0016E5_07975.png
     │   └─ ...
     ├─ valannot
     │   ├─ 0016E5_07959.png
     │   ├─ 0016E5_07975.png
     │   └─ ...
     │
     ├─ test.txt
     ├─ test
     │   ├─ 0001TP_008550.png
     │   ├─ 0001TP_008910.png
     │   └─ ...
     └─ testannot
         ├─ 0001TP_008550.png
         ├─ 0001TP_008910.png
         └─ ... 
```
Cityscapes Dataset : [Cityscapes](https://www.cityscapes-dataset.com/)
The directory structure of this dataset is shown below. In the CamVid dataset, both training and annotation data are binary image files.
```Shell
    Cityscapes
     ├─ gtfine
     │   ├─ test
     │   │    ├─ <cityfolder>
     │   │    └─ <cityfolder>        
     │   ├─ train
     │   │    ├─ <cityfolder>
     │   │    └─ <cityfolder>         
     │   └─ val
     │        ├─ <cityfolder>
     │        └─ <cityfolder>   
     └─ leftImg8bit
         ├─ test
         │    ├─ <cityfolder>
         │    └─ <cityfolder>        
         ├─ train
         │    ├─ <cityfolder>
         │    └─ <cityfolder>         
         └─ val
              ├─ <cityfolder>
              └─ <cityfolder> 
```
## Usage
Run [``main.py``](https://github.com/w11m/03_ENet_kd/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [--submode] [--modelname] # Vanilla Setting
               [--kdmethod] [--pimode] [--teacher_model] [--teacher_dir] [--student_model] # KD Setting
               [--mutual_model_num] [--mutual_models] [--mutualpimode] # mutual setting 
```
### Training Vanilla Model
Example training script can be find in [``train_camvid_vanilla.sh``](https://github.com/w11m/03_ENet_kd/blob/master/script/train_camvid_vanilla.sh)
```
python main.py --submode 'vanilla' --modelname <MODEL_NAME>
```

### Training KD Model
Example training script can be find in [``train_camvid_kd.sh``](https://github.com/w11m/03_ENet_kd/blob/master/script/train_camvid_kd.sh)
```
python main.py --submode 'kd' --teacher_model <TMODEL_NAME> --teacher_dir <TMODEL_PATH> --student_model <SMODEL_NAME>
```

### Training Mutual Model
Example training script can be find in [``train_camvid_mutual.sh``](https://github.com/w11m/03_ENet_kd/blob/master/script/train_camvid_mutual.sh)
```
python main.py --submode 'mutual' --mutual_models <MODEL_NAME><MODEL_NAME>
```

## File structure
``argument.py``: For argument setting, 'train_mode' 'epoch' 'learning_rate' etc.

``mytransform.py``: For Image type transform 'PILtoTENSOR' or 'TENSORtoPIL'

``criterion.py``: For defining knowledge distillation loss.

``myutlis.py``: For code init example: make directory, print loss, tensorboard writer.

``trainer.py``: Training code. ``tester.py``: Testing (Validation and Test) code.

``main.py``: Main program, it will use ``trainer`` and ``tester`` for training.

``check_fps.py``: Use fake tensor for model inference speed test.

``submit_cityscapes.py``: For individually test `Citscapes` dataset. 

``visualize_numpy.py``: For visualize demo. 


