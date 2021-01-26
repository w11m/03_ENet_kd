import torch
from model.model_selector import ModelSelector
import torch.nn as nn
from metric.iou import IoU

class Tester:
    def __init__(self, data_loader, classweight,classencoding, args):
        self.data_loader = data_loader
        self.mutual_model_num = args.mutual_model_num
        self.metricS = []
        self.device = args.device
        self.ce_criterion = nn.CrossEntropyLoss(weight=classweight)
        self.args = args
        if args.ignore_unlabeled:
            self.ignore_index = list(classencoding).index('unlabeled')
        else:
            self.ignore_index = None
        self.loop = 1
        if self.args.submode == 'mutual':
            self.loop = self.mutual_model_num
        for i in range(self.loop):
            exec("metric{} = IoU(args.numclass, ignore_index=self.ignore_index)".format(i))
            exec("self.metricS.append(metric{})".format(i))
    def run_epoch(self, modelS):
        epoch_loss = [0.] * self.loop
        for i in range(self.loop):
            modelS[i].eval()
            self.metricS[i].reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            outputS = []
            with torch.no_grad():
                # Forward propagation
                for i in range(self.loop):
                    outputS.append(modelS[i](inputs))
                    # Loss computation
                    loss = self.ce_criterion(outputS[i], labels)
                    epoch_loss[i] += loss.item()
                    self.metricS[i].add(outputS[i].detach(), labels.detach())
        for i in range(self.loop):
            epoch_loss[i] / len(self.data_loader)
        return epoch_loss, self.metricS

class Outside_Tester:
    def __init__(self, data_loader, classweight,classencoding, args):
        self.data_loader = data_loader
        self.mutual_model_num = args.mutual_model_num
        self.metricS = []
        self.device = args.device
        self.ce_criterion = nn.CrossEntropyLoss(weight=classweight)
        self.args = args
        if args.ignore_unlabeled:
            self.ignore_index = list(classencoding).index('unlabeled')
        else:
            self.ignore_index = None
        self.loop = 1
        if self.args.submode == 'mutual':
            self.loop = self.mutual_model_num
        for i in range(self.loop):
            exec("metric{} = IoU(args.numclass, ignore_index=self.ignore_index)".format(i))
            exec("self.metricS.append(metric{})".format(i))
    def select_run_test(self,modelS):
        if self.args.dataset == 'CamVid':
            return self.run_camvid_test(modelS)
        elif self.args.dataset == 'Cityscapes':
            return self.run_city_test(modelS)

    def run_camvid_test(self,modelS):
        import os
        epoch_loss = [0.] * self.loop
        for i in range(self.loop):
            modelS[i].eval()
            self.metricS[i].reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            outputS = []
            with torch.no_grad():
                # Forward propagation
                for i in range(self.loop):
                    outputS.append(modelS[i](inputs))
                    # Loss computation
                    loss = self.ce_criterion(outputS[i], labels)
                    epoch_loss[i] += loss.item()
                    self.metricS[i].add(outputS[i].detach(), labels.detach())
        for i in range(self.loop):
            epoch_loss[i] / len(self.data_loader)
        for i in range(self.loop):
            summary_filename = os.path.join(self.args.save_dir, 'ModelBestInfo_{}.txt'.format(i))
            with open(summary_filename, 'a') as summary_file:
                summary_file.write("\n[BEST TESTING]")
                summary_file.write("\nMean IoU: {0}".format(self.metricS[i].value()[1]))
            summary_file.close()
            print("\n[Best model {}] Testing".format(i))
            print("[Avg.loss]: {0:.4f} | Mean IoU: {1:.4f}".format(epoch_loss[i], self.metricS[i].value()[1]))

    def run_city_test(self, modelS):
        import numpy as np
        import torch.nn.functional as F
        import os
        import cv2
        trainId_to_id = {
            0: 7, 1: 8, 2: 11, 3: 12, 4: 13,
            5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
            10: 23, 11: 24, 12: 25, 13: 26, 14: 27,
            15: 28, 16: 31, 17: 32, 18: 33, 19: 0
        }
        trainId_to_id_map_func = np.vectorize(trainId_to_id.get)
        for i in range(self.loop):
            modelS[i].eval()
            mutual_submit_dir = self.args.save_dir + '/citycapes_submmit/' + '{}/'.format(i)
            try:
                os.makedirs(mutual_submit_dir)
            except:
                pass
            for step, (batch, img_path) in enumerate(self.data_loader):
                with torch.no_grad():
                    img = batch.to('cuda')
                    imgPT = img_path
                    outputs = modelS[i](img)
                    outputs = F.interpolate(outputs, size=(1024, 2048), mode="bilinear")
                    _, outputs = torch.max(outputs.data, 1)
                    for i in range(outputs.size(0)):
                        pred_label_img = outputs.cpu().numpy()[i]
                        pred_label_img = pred_label_img - 1
                        pred_label_img.astype(np.uint8)
                        img_id = imgPT[i]
                        pred_label_img = trainId_to_id_map_func(pred_label_img)  # (shape: (1024, 2048))
                        pred_label_img = pred_label_img.astype(np.uint8)

                        cv2.imwrite(mutual_submit_dir + "/" + img_id + "_pred_label_img.png", pred_label_img)
