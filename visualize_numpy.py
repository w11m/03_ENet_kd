import sys, os
import torch
import torch.nn
import os.path as osp
import datetime
from arguments import ArgumentParser
from PIL import Image
from model.model_selector import ModelSelector
import torchvision.transforms as transforms
import time
import re
import numpy as np
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
    ar.save_dir = osp.join(ar.save_dir, ar.model)
    try:
        os.makedirs(ar.save_dir)
    except:
        pass
    folder_num = len(next(os.walk(ar.save_dir))[1])
    folder_name = osp.join(ar.save_dir, ar.mo)
    os.makedirs(folder_name)
    ar.save_dir = folder_name

def demo(args, color_encoding):
    pathdir_init(args)
    selector = ModelSelector(len(color_encoding))
    model = selector.segmentor(args.model).to(args.device)
    weightfile = torch.load(args.model_path)
    model.load_state_dict((weightfile['state_dict']))

    image_transform = transforms.Compose([transforms.Resize((args.height, args.width)), transforms.ToTensor()])
    pic_list = os.listdir(args.demo_dir)
    gt_list = os.listdir(args.demo_gt)

    pic_list = natural_sort(pic_list)
    gt_list = natural_sort(gt_list)
    pic_num = len(pic_list)
    allfps = 0
    ticsys = time.time()

    for index, imgpath in enumerate(pic_list):
        cv2pic = cv2.imread(osp.join(args.demo_dir, imgpath))
        pilpic = Image.fromarray(cv2.cvtColor(cv2pic, cv2.COLOR_BGR2RGB))

        pic_tensor = image_transform(pilpic)
        pic_tensor = torch.unsqueeze(pic_tensor, 0).to(args.device)
        with torch.no_grad():
            tic = time.time()
            predict = model(pic_tensor)
            toc = time.time()
            tttime = toc - tic
            fps = 1 / tttime
            print('loop{}:{}'.format(index, fps))
            allfps = allfps + fps
            _, predict = torch.max(predict.data, 1)
            predict = predict.cpu().numpy()[0]
            if args.dataset == 'cityscapes':
                predict = predict - 1
            mask_color = np.asarray(color_encoding[predict], dtype=np.uint8)
            mask_color = cv2.resize(mask_color, (args.width_real, args.height_real),
                                    interpolation=cv2.INTER_NEAREST)
            mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
            cv2pic = cv2.resize(cv2pic, (args.width_real, args.height_real))
            res = cv2.addWeighted(cv2pic, 0.5, mask_color, 0.6, 1)
            cv2.imshow('res', res)
            cv2.imshow('Mask', mask_color)
            cv2.waitKey(1)
            cv2.imwrite(osp.join(args.save_dir, 'res_{}.png'.format(index)), res)
            cv2.imwrite(osp.join(args.save_dir, 'mask_color{}.png'.format(index)), mask_color)
    # for index, imgpath in enumerate(gt_list):
    #     cv2pic = cv2.imread(osp.join(args.demo_gt, imgpath),cv2.IMREAD_GRAYSCALE)
    #     pilpic = Image.fromarray(cv2.cvtColor(cv2pic, cv2.COLOR_BGR2RGB))
    #
    #     mask_color = np.asarray(color_encoding[cv2pic], dtype=np.uint8)
    #     mask_color = cv2.resize(mask_color, (args.width_real, args.height_real),
    #                                 interpolation=cv2.INTER_NEAREST)
    #     mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    #     cv2pic = cv2.resize(cv2pic, (args.width_real, args.height_real))
    #     cv2.imshow('GT', mask_color)
    #     cv2.waitKey(1)
    #     cv2.imwrite(osp.join(args.save_dir, 'GT{}.png'.format(index)), mask_color)
    tocsys = time.time()
    fps_avg = allfps / pic_num
    print('inference avg_fps:{}'.format(fps_avg))
    print('system avg_fps:{}'.format((pic_num) / (tocsys - ticsys)))

if __name__ == '__main__':
    visulparser = ArgumentParser()

    # Execution mode
    visulparser.add_argument("--model", default='ENet_2enc0.5')

    visulparser.add_argument("--model_path", default='/media/tan/Disk/graduate_Code/ENet_ncist/0108reDU_CE_TEST/vanilla/CamVid/CE/ENet_2enc0.5/0/model0_BEST_ckpt.pth')
    visulparser.add_argument("--mo",default='')

    visulparser.add_argument("--demo_dir", default='./DemoPic/Camvid')
    visulparser.add_argument("--demo_gt", default='./demo_gt')

    # Hyperparameters
    visulparser.add_argument("--dataset", choices=['camvid', 'cityscapes', 'demo'], default='camvid')

    visulparser.add_argument("--height", type=int, default=360, help="512")

    visulparser.add_argument("--width", type=int, default=480, help="512")

    visulparser.add_argument("--resize", type=str2bool, default=True)

    visulparser.add_argument("--height_real", type=int, default=360, help="The image height. Default: 360")

    visulparser.add_argument("--width_real", type=int, default=480, help="The image width. Default: 480")

    # Settings
    visulparser.add_argument("--device", default='cuda')

    # Storage settings
    visulparser.add_argument("--save_dir", type=str, default='./demo_result/')

    args = visulparser.parse_args()

    if args.dataset == 'camvid':

        color_index = ['sky', 'building', 'pole', 'road',
                       'pavement', 'tree', 'sign_symbol', 'fence',
                       'car', 'pedestrian', 'bicyclist', 'unlabeled']
        color_encoding = np.asarray([[128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
                                     [60, 40, 222], [128, 128, 0], [192, 128, 128], [64, 64, 128],
                                     [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0],
                                     ])
    elif args.dataset == 'cityscapes':
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
    demo(args, color_encoding)
