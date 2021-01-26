from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ('Unsupported value encountered.')


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("--mode", choices=['train'], default='train')
    parser.add_argument("--submode", choices=['vanilla', 'kd', 'mutual'],default='mutual')

    # Vanilla Setting
    parser.add_argument("--modelname", default='ENet_slim0.5')

    # KD Setting
    parser.add_argument("--kdmethod", nargs='+', default=['pixelwise','pairwise'])
    parser.add_argument("--pimode", default='KL')
    parser.add_argument("--teacher_model", default='ENet')
    parser.add_argument("--teacher_dir", type=str,
                        default='/media/tan/Disk/graduate_Code/ENet_ncist/camvid360480_saved_ckpt/2020-12-10/2020-12-10_0/model0_ckpt.pth')
    parser.add_argument("--student_model", default='ENet_3enc0_channel0.6')
    # Mutual KD Setting

    parser.add_argument("--mutual_model_num", type=int, default=2)
    parser.add_argument("--mutual_models", nargs='+', default=['ENet_3enc0_channel0.6', 'ENet'])
    parser.add_argument("--mutualpimode", default='KL')
    parser.add_argument("--tensorboard", default=True, type=str2bool)
    parser.add_argument("--seeds", default=2018, type=int)
    parser.add_argument("--loop",default=1) # don't need to change

    # General setting
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    # Learning rate updater
    parser.add_argument("--lr_update", choices=['StepLR', 'ReduceLROnPlateau'], default='ReduceLROnPlateau')

    # Optimizer setting
    parser.add_argument("--lr", type=float, default=5e-4) # initial learning rate
    parser.add_argument("--weightdecay", type=float, default=2e-4)

    # StepLR_setting
    parser.add_argument("--St_lrdecay", type=float, default=0.1) # lr_decay for StepLR
    parser.add_argument("--St_lrdecay_epochs", type=int, default=100)

    # ReduceLROnPlateau_setting
    parser.add_argument("--Re_mode", type=str, default='max')
    parser.add_argument("--Re_factor", type=float, default=0.6)
    parser.add_argument("--Re_patience", type=int, default=1)
    parser.add_argument("--Re_min_lr", type=float, default=5e-6)

    # Dataset
    parser.add_argument("--dataset", choices=['CamVid', 'Cityscapes'], default='CamVid')
    parser.add_argument("--dataset_dir", type=str, default="./CamVid")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--weighing", choices=['enet', 'mfb', 'none'], default='enet')
    parser.add_argument("--with-unlabeled", dest='ignore_unlabeled', action='store_false')

    # Storage settings
    parser.add_argument("--name", type=str, default='model', help="Name given to the model when saving. Default: ENet")
    parser.add_argument("--save_dir", type=str, default='training_result')
    parser.add_argument("--save_step",type=int,default=2)
    parser.add_argument("--weighing_dir", type=str, default='weighing_file')

    # Settings
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default='cuda')

    return parser.parse_args()
