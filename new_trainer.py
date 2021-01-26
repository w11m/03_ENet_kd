from model.model_selector import ModelSelector
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import myutils
from criterion import CriterionCEPixelWise
from criterion import CriterionJSDPixelWise
from criterion import CriterionKLPixelWise
from criterion import CriterionPairWise
from criterion import CriterionFeatureCorrelation
from metric.iou import IoU

class Trainer(object):
    def __init__(self, data_loader, classweight, classencoding, args):
        self.data_loader = data_loader
        self.modelS = []
        self.optimizerS = []
        self.schedulerS = []
        self.metricS = []
        self.mutualpimode = args.mutualpimode
        self.pimode = args.pimode
        self.device = args.device
        self.ce_criterion = nn.CrossEntropyLoss(weight=classweight)
        self.selector = ModelSelector(args.numclass)
        self.args = args
        self.mutual_model_num = args.mutual_model_num
        if args.ignore_unlabeled:
            self.ignore_index = list(classencoding).index('unlabeled')
        else:
            self.ignore_index = None
        if args.submode == 'mutual':
            if len(args.mutual_models) == args.mutual_model_num:
                different = True
            else:
                different = False
                print("[Warning] please check argument parser setting")
                print("Mutual model number {} doesn't fit {}".format(args.mutual_model_num, args.mutual_models))
                print("Set {} args.mutual_models[0] as only mutual learning model".format(args.mutual_models[0]))
            for i in range(self.mutual_model_num):
                if different:
                    model = self.selector.segmentor(args.mutual_models[i]).to(self.device)
                elif i == 0 and not different:
                    model = self.selector.segmentor(args.mutual_models[0]).to(self.device)
                self.modelS.append(model)
                exec("metric{} = IoU(args.numclass, ignore_index=self.ignore_index)".format(i))
                exec("self.metricS.append(metric{})".format(i))
                exec("optimizer{} = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)".format(i))
                exec("self.optimizerS.append(optimizer{})".format(i))
                if args.lr_update == 'StepLR':
                    reduce_lr = "lr_updater{} = lr_scheduler.StepLR(optimizer{}, {}, {},verbose=True)"
                    reduce_lr = reduce_lr.format(i, i, args.St_lrdecay_epochs, args.St_lrdecay)
                    exec(reduce_lr)
                elif args.lr_update == 'ReduceLROnPlateau':
                    reduce_lr = "lr_updater{} = lr_scheduler.ReduceLROnPlateau(optimizer{}, mode='{}', factor={}, patience={}, min_lr={},verbose=True)"
                    reduce_lr = reduce_lr.format(i, i, args.Re_mode, args.Re_factor, args.Re_patience, args.Re_min_lr)
                    exec(reduce_lr)
                exec("self.schedulerS.append(lr_updater{})".format(i))
            for i in range(self.mutual_model_num):
                paramsg = '[*] Number of parameters of {} model: {:,}'
                print(paramsg.format(args.mutual_models[i],
                                     sum([p.data.nelement() for p in self.modelS[i].parameters()])))
            if args.mutualpimode == 'KL':
                self.criterion_pixelwise = CriterionKLPixelWise()
            elif args.mutualpimode == "JSD":
                self.criterion_pixelwise = CriterionJSDPixelWise()
            elif args.mutualpimode == "CE":
                self.criterion_pixelwise = CriterionCEPixelWise()
        elif args.submode == 'kd':
            i = 0
            teacher_model = self.selector.segmentor(args.teacher_model).to(self.device)
            try:
                teacher_model = myutils.teacher_loader(teacher_model, args)
            except:
                pass
            student_model = self.selector.segmentor(args.student_model).to(self.device)
            self.modelS.append(student_model)
            self.modelS.append(teacher_model)
            exec("metric{} = IoU(args.numclass, ignore_index=self.ignore_index)".format(i))
            exec("self.metricS.append(metric{})".format(i))
            opt = "optimizer{} = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weightdecay)"
            opt = opt.format(i)
            exec(opt)
            exec("self.optimizerS.append(optimizer{})".format(i))
            if args.lr_update == 'StepLR':
                reduce_lr = "lr_updater{} = lr_scheduler.StepLR(optimizer{}, {}, {},verbose=True)"
                reduce_lr = reduce_lr.format(i, i, args.St_lrdecay_epochs, args.St_lrdecay)
                exec(reduce_lr)
            elif args.lr_update == 'ReduceLROnPlateau':
                reduce_lr = "lr_updater{} = lr_scheduler.ReduceLROnPlateau(optimizer{}, mode='{}', factor={}, patience={}, min_lr={},verbose=True)"
                reduce_lr = reduce_lr.format(i, i, args.Re_mode, args.Re_factor, args.Re_patience, args.Re_min_lr)
                exec(reduce_lr)
            exec("self.schedulerS.append(lr_updater{})".format(i))
            paramsg_t = '[*] Number of parameters of {} teacher model: {:,}'
            paramsg_s = '[*] Number of parameters of {} student model: {:,}'
            print(paramsg_t.format(args.teacher_model, sum([p.data.nelement() for p in self.modelS[1].parameters()])))
            print(paramsg_s.format(args.student_model, sum([p.data.nelement() for p in self.modelS[0].parameters()])))
            if args.pimode == 'KL':
                self.criterion_pixelwise = CriterionKLPixelWise()
            elif args.pimode == "JSD":
                self.criterion_pixelwise = CriterionJSDPixelWise()
            elif args.pimode == "CE":
                self.criterion_pixelwise = CriterionCEPixelWise()
            self.criterion_pairwise = CriterionPairWise()
            self.criterion_featurecorrelation = CriterionFeatureCorrelation(poolsize=15)


        elif args.submode == 'vanilla':
            i = 0
            model = self.selector.segmentor(args.modelname).to(self.device)
            self.modelS.append(model)
            exec("metric{} = IoU(args.numclass, ignore_index=self.ignore_index)".format(i))
            exec("self.metricS.append(metric{})".format(i))
            opt = "optimizer{} = optim.Adam(self.modelS[0].parameters(), lr=args.lr, weight_decay=args.weightdecay)"
            opt = opt.format(i)
            exec(opt)
            exec("self.optimizerS.append(optimizer{})".format(i))
            if args.lr_update == 'StepLR':
                reduce_lr = "lr_updater{} = lr_scheduler.StepLR(optimizer{}, {}, {},verbose=True)"
                reduce_lr = reduce_lr.format(i, i, args.St_lrdecay_epochs, args.St_lrdecay)
                exec(reduce_lr)
            elif args.lr_update == 'ReduceLROnPlateau':
                reduce_lr = "lr_updater{} = lr_scheduler.ReduceLROnPlateau(optimizer{}, mode='{}', factor={}, patience={}, min_lr={},verbose=True)"
                reduce_lr = reduce_lr.format(i, i, args.Re_mode, args.Re_factor, args.Re_patience, args.Re_min_lr)
                exec(reduce_lr)
            exec("self.schedulerS.append(lr_updater{})".format(i))
            paramsg_v = '[*] Number of parameters of {} vanilla model: {:,}'
            print(paramsg_v.format(args.student_model, sum([p.data.nelement() for p in self.modelS[0].parameters()])))
    def select_run_epoch(self):
        if self.args.submode == 'vanilla':
            return self.run_vanilla_one_epoch()
        elif self.args.submode == 'kd':
            return self.run_kd_one_epoch()
        elif self.args.submode == 'mutual':
            return self.run_mutual_one_epoch()

    def run_mutual_one_epoch(self):
        ce_lossS = [0.0] * self.mutual_model_num
        pixel_lossS = [0.0] * self.mutual_model_num
        for i in range(self.mutual_model_num):
            self.modelS[i].train()
            self.metricS[i].reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            outputs = []
            for model in self.modelS:
                outputs.append(model(inputs))
            for i in range(self.mutual_model_num):
                ce_loss = self.ce_criterion(outputs[i], labels)
                pixel_loss = 0
                ce_lossS[i] += ce_loss.item()
                for j in range(self.mutual_model_num):
                    # print("Loop{}/{}".format(i,j))
                    if i != j:
                        pixel_loss = self.criterion_pixelwise(outputs[i], (outputs[j].detach()))
                        pixel_lossS[i] += pixel_loss.item()
                loss = ce_loss + pixel_loss / (self.mutual_model_num - 1)
                self.optimizerS[i].zero_grad()
                loss.backward()
                self.optimizerS[i].step()
                self.metricS[i].add(outputs[i].detach(), labels.detach())
        for i in range(self.mutual_model_num):
            ce_lossS[i] / len(self.data_loader)
            pixel_lossS[i] / len(self.data_loader)
            self.metricS[i].value()
            if self.args.lr_update == 'StepLR':
                self.schedulerS[i].step()
        return [ce_lossS, pixel_lossS], self.metricS, self.modelS, self.optimizerS, self.schedulerS

    def run_kd_one_epoch(self):
        self.modelS[1].eval()  # teacher model eval
        self.modelS[0].train()  # student model train
        epoch_celoss = [0.0]
        epoch_piloss = [0.0]
        epoch_paloss = [0.0]
        epoch_corloss = [0.0]
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            # Forward propagation
            outputs_student = self.modelS[0](inputs)
            with torch.no_grad():
                outputs_teacher = self.modelS[1](inputs)
            piloss = 0
            paloss = 0
            corloss = 0
            # Loss computation
            celoss = self.ce_criterion(outputs_student, labels)
            if 'pixelwise' in self.args.kdmethod:
                piloss = self.criterion_pixelwise(outputs_student, outputs_teacher)
            if 'pairwise' in self.args.kdmethod:
                paloss = self.criterion_pairwise(outputs_student, outputs_teacher)
            if 'correlation' in self.args.kdmethod:
                std_mid_feature = self.modelS[0].feature1
                tcr_mid_feature = self.modelS[1].feature1
                corloss = self.criterion_featurecorrelation(std_mid_feature, tcr_mid_feature)
            loss = celoss + piloss + paloss + corloss
            # Backpropagation
            self.optimizerS[0].zero_grad()
            loss.backward()
            self.optimizerS[0].step()
            # Keep track of loss for current epoch
            epoch_piloss[0] += piloss
            epoch_paloss[0] += paloss
            epoch_corloss[0] += corloss
            epoch_celoss[0] += celoss

            # Keep track of the evaluation metric
            self.metricS[0].add(outputs_student.detach(), labels.detach())
        epoch_celoss[0] = epoch_celoss[0] / len(self.data_loader)
        epoch_piloss[0] = epoch_piloss[0] / len(self.data_loader)
        epoch_paloss[0]= epoch_paloss[0] / len(self.data_loader)
        epoch_corloss[0] = epoch_corloss[0] / len(self.data_loader)
        if self.args.lr_update == 'StepLR':
            self.schedulerS[0].step()
        return [epoch_celoss, epoch_piloss, epoch_paloss,
                epoch_corloss], self.metricS, self.modelS, self.optimizerS, self.schedulerS

    def run_vanilla_one_epoch(self):
        self.modelS[0].train()
        epoch_celoss = [0.0]
        self.metricS[0].reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            outputs = self.modelS[0](inputs)
            # Loss computation
            loss = self.ce_criterion(outputs, labels)

            # Backpropagation
            self.optimizerS[0].zero_grad()
            loss.backward()
            self.optimizerS[0].step()

            # Keep track of loss for current epoch
            epoch_celoss[0] += loss.item()

            # Keep track of the evaluation metric
            self.metricS[0].add(outputs.detach(), labels.detach())
        epoch_celoss[0] = epoch_celoss[0] / len(self.data_loader)
        if self.args.lr_update == 'StepLR':
            self.schedulerS[0].step()
        return [epoch_celoss], self.metricS, self.modelS, self.optimizerS, self.schedulerS
