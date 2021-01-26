import pytorch_ssim
import torch
from torch.autograd import Variable
import argparse
from torchvision import transforms as T
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.structures.image_list import to_image_list
import copy
import os
import cv2
from demo.predictor import Resize
from torch import nn



image = '/home/lab70636/yuchieh/centermask_learn/demo/images/COCO_val2014_000000463842.jpg'
ori_img = cv2.imread(image)
device = torch.device(cfg.MODEL.DEVICE)


def hook_s(module, input, output):
    s_outputs.append(output)


def hook_t(module, input, output):
    t_outputs.append(output)


def build_transform(cfg):

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transform = T.Compose(
        [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform


def model(cfg):
    model = build_detection_model(cfg)
    model.to(device)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    return model

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

    parser.add_argument(
        "--teacher-config-file",
        default="../configs/centermask/centermask_V_19_eSE_FPN_ms_3x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--student-config-file",
        default="../configs/centermask/centermask_V_19_eSE_FPN_lite_res600_ms_bs16_4x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    global args
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    t_cfg = copy.deepcopy(cfg)

    cfg.merge_from_file(args.student_config_file)
    cfg.freeze()

    t_cfg.merge_from_file(args.teacher_config_file)
    t_cfg.freeze()

    s_model = model(cfg)
    s_model.eval()
    t_model = model(t_cfg)
    t_model.eval()

    transform = build_transform(cfg)
    img = transform(ori_img)
    image_list = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(device)

    global s_outputs
    global t_outputs
    s_outputs = []
    t_outputs = []
    with torch.no_grad():
        handle_s1 = s_model.backbone.body.stage2.OSA2_1.ese.hsigmoid.register_forward_hook(hook_s)
        handle_s2 = s_model.backbone.body.stage2.OSA2_1.concat[2].register_forward_hook(hook_s)
        s_model(image_list)
        handle_s1.remove()
        handle_s2.remove()
        s_att = s_outputs[1] * s_outputs[0]

        handle_t1 = t_model.backbone.body.stage2.OSA2_1.ese.hsigmoid.register_forward_hook(hook_t)
        handle_t2 = t_model.backbone.body.stage2.OSA2_1.concat[2].register_forward_hook(hook_t)
        t_model(image_list)
        handle_t1.remove()
        handle_t2.remove()
        t_att = t_outputs[1] * t_outputs[0]

    MSE = nn.MSELoss()(s_att, t_att)
    print('MSE: {}'.format(MSE))


    img_s = Variable(s_att)
    img_t = Variable(t_att)

    if torch.cuda.is_available():
        img_s = img_s.cuda()
        img_t = img_t.cuda()

    print(pytorch_ssim.ssim(img_s, img_t))

    # ssim_loss = pytorch_ssim.SSIM(window_size=11)

    # print(ssim_loss(img_s, img_t))

if __name__ == "__main__":
    main()
