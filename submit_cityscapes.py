import sys, os
import torch
import torch.nn
import os.path as osp
import datetime
from arguments import ArgumentParser
from PIL import Image
from model.model_selector import ModelSelector
from data.get_datasets import select_dataset
from data.get_datasets import load_submit_dataset
import torchvision.transforms as transforms
import time
import re
import numpy as np
import torch.nn.functional as F
import cv2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ('Unsupported value encountered.')


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def pathdir_init(ar):
    ar.save_dir = osp.join(ar.save_dir, str(datetime.date.today()))
    try:
        os.makedirs(ar.save_dir)
    except:
        pass
    folder_num = len(next(os.walk(ar.save_dir))[1])
    folder_name = osp.join(ar.save_dir, str(datetime.date.today()) + '_' + str(folder_num))
    os.makedirs(folder_name)
    ar.save_dir = folder_name


def eval_to_submit(args, color_encoding):
    pathdir_init(args)
    dataset = select_dataset(set_name=args.dataset)
    dataload_, class_weights, class_encoding = load_submit_dataset(args, dataset)
    selector = ModelSelector(len(class_encoding))
    model = selector.segmentor(args.model).to(args.device)
    weightfile = torch.load(args.model_path)
    model.load_state_dict((weightfile['state_dict']))

    image_transform = transforms.Compose([transforms.Resize((args.height, args.width)), transforms.ToTensor()])
    count = 0
    total = 1525
    for step, (batch, img_path) in enumerate(dataload_[1]):
        with torch.no_grad():
            img = batch.to('cuda')
            imgPT = img_path
            outputs = model(img)
            outputs = F.interpolate(outputs, size=(1024, 2048), mode="bilinear")
            _, outputs = torch.max(outputs.data, 1)
            count += outputs.size(0)
            print("{}/{}".format(count, total))
            for i in range(outputs.size(0)):
                pred_label_img = outputs.cpu().numpy()[i]
                pred_label_img = pred_label_img - 1
                pred_label_img.astype(np.uint8)
                img_id = imgPT[i]
                pred_label_img = trainId_to_id_map_func(pred_label_img)  # (shape: (1024, 2048))
                pred_label_img = pred_label_img.astype(np.uint8)

                cv2.imwrite(args.save_dir + "/" + img_id + "_pred_label_img.png", pred_label_img)


if __name__ == '__main__':
    visulparser = ArgumentParser()

    # Execution mode
    visulparser.add_argument("--model", default='ENet')

    visulparser.add_argument("--model_path", default='/media/tan/Samsung_T5/1024x2048/ENet_vanilla.pth')

    # Hyperparameters
    visulparser.add_argument("--dataset", choices=['Cityscapes'], default='Cityscapes')

    visulparser.add_argument("--dataset_dir", type=str, default="./Cityscapes")

    visulparser.add_argument("--height", type=int, default=512, help="512")

    visulparser.add_argument("--width", type=int, default=1024, help="1024")

    visulparser.add_argument("--resize", type=str2bool, default=True)

    visulparser.add_argument("--batch_size", type=int, default=10)
    visulparser.add_argument("--workers", type=int, default=4)

    visulparser.add_argument("--height_real", type=int, default=1024, help="The image height. Default: 360")

    visulparser.add_argument("--width_real", type=int, default=2048, help="The image width. Default: 480")
    # Settings
    visulparser.add_argument("--device", default='cuda')
    visulparser.add_argument("--weighing", choices=['enet', 'mfb', 'none'], default='enet')
    visulparser.add_argument("--with-unlabeled", dest='ignore_unlabeled', action='store_false')
    visulparser.add_argument("--weighing_dir", type=str, default='weighing_file')
    # Storage settings
    visulparser.add_argument("--save_dir", type=str, default='./submit_result/Vanilla_ENet/')

    args = visulparser.parse_args()

    color_encoding_index = ['road', 'sidewalk', 'building',
                            'wall', 'fence', 'pole', 'traffic_light',
                            'traffic_sign', 'vegetation', 'terrain', 'sky',
                            'person', 'rider', 'car', 'truck',
                            'bus', 'train', 'motorcycle', 'bicycle', 'unlabeled']
    color_encoding = np.asarray([
        [128, 64, 128], [244, 35, 232], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
        [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
    ])

    trainId_to_id = {
        0: 7, 1: 8, 2: 11, 3: 12, 4: 13,
        5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
        10: 23, 11: 24, 12: 25, 13: 26, 14: 27,
        15: 28, 16: 31, 17: 32, 18: 33, 19: 0
    }
    trainId_to_id_map_func = np.vectorize(trainId_to_id.get)
    eval_to_submit(args, trainId_to_id_map_func)
