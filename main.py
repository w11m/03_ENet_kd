import torch
import os, sys
import random
import numpy as np
from arguments import get_arguments
import myutils
import os.path as osp
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from data.get_datasets import select_dataset
from data.get_datasets import load_dataset
from trainer import Trainer
from tester import Tester
from tester import Outside_Tester
import myutils

if __name__ == '__main__':
    args = myutils.code_init()
    dataset = select_dataset(set_name=args.dataset)
    dataload_, class_weights, class_encoding = load_dataset(args, dataset)
    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.save_dir)
    else:
        writer = None
    train = Trainer(dataload_[0], class_weights, class_encoding, args)
    val = Tester(dataload_[1], class_weights, class_encoding, args)
    test = Outside_Tester(dataload_[2], class_weights, class_encoding, args)
    best_miou = [0.] * args.loop
    for epoch in range(0, args.epochs):
        epoch_lossS, train_IoUS, modelS, optimS, schedulerS = train.select_run_epoch()
        myutils.train_loss_printer(writer,epoch_lossS,train_IoUS,optimS,epoch,args)
        lossS, val_IoUS = val.run_epoch(modelS)
        if args.lr_update == 'ReduceLROnPlateau':
            for i in range(args.loop):
                schedulerS[i].step(val_IoUS[i].value()[1])
        save = myutils.val_loss_printer(writer, lossS, val_IoUS, epoch, args,best_miou)
        myutils.save_checkpoint(save,modelS,optimS,val_IoUS,epoch,args)
    if args.tensorboard:
        writer.close()

    for i in range(args.loop):
        model_path = os.path.join(args.save_dir, args.name + str(i) + '_BEST_ckpt' + '.pth')
        checkpoint = torch.load(model_path)
        modelS[i].load_state_dict(checkpoint['state_dict'])
    test.select_run_test(modelS)