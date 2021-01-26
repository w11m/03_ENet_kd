# ENet_kd
this is an ENet knowledge distillation

## Introduction
This code works for semantic segmentation (ENet) knowledge distillation 

The master branch works with **Pytorch 1.7.0**, with **Python 3.5,3.6**.

### Training Data

This implementation has been tested on the CamVid and Cityscapes datasets.

CamVid Dataset : [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

Cityscapes Dataset : [Cityscapes](https://www.cityscapes-dataset.com/)

## Usage

Run [``main.py``](https://github.com/w11m/03_ENet_kd/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [--submode] [--modelname] # Vanilla Setting
               [--kdmethod] [--pimode] [--teacher_model] [--teacher_dir] [--student_model] # KD Setting
               [--mutual_model_num] [--mutual_models] [--mutualpimode] # mutual setting 
```

For help on the optional arguments run: ``python main.py -h``