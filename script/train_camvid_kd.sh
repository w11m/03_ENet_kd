## 'StepLR' training
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.75' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.5' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.25' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_2enc0.5_3enc0.5' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_2enc0.5' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0_channel0.75' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0_channel0.6' --lr_update 'StepLR' --dataset 'CamVid' --dataset_dir './CamVid'

## 'ReduceLROnPlateau' training
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.75' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.5' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_slim0.25' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_2enc0.5_3enc0.5' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_2enc0.5' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0_channel0.75' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'
python3 main.py --submode 'kd' --teacher_model 'ENet' --teacher_dir '/<teacher_path>.pth' --student_model 'ENet_3enc0_channel0.6' --lr_update 'ReduceLROnPlateau' --dataset 'CamVid' --dataset_dir './CamVid'

