import torch
import numpy as np
import random
import sys, os
import os.path as osp
from arguments import get_arguments

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.log.write("Python Version:{}.{}\n".format(sys.version_info.major, sys.version_info.minor))
        self.log.write("Torch Version:{}\n".format(torch.__version__))
        self.log.write("Cudnn Version:{}\n\n".format(torch.backends.cudnn.version()))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def teacher_loader(model, args):
    weight_path = args.teacher_dir
    assert os.path.isfile(weight_path), "The model file \"{0}\" doesn't exist.".format(weight_path)
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def pathdir_init(ar):
    if ar.submode == 'vanilla':
        ar.save_dir = osp.join(ar.save_dir, ar.submode, ar.dataset, 'CE', ar.modelname.replace('.', '_'))
    elif 'kd' in ar.submode:
        kd_mode = ''.join(str(e) for e in ar.kdmethod)
        ar.save_dir = osp.join(ar.save_dir, ar.submode, ar.dataset, kd_mode + '_' + ar.pimode, ar.student_model.replace('.', '_'))
    elif 'mutual' in ar.submode:
        ar.save_dir = osp.join(ar.save_dir, ar.submode, ar.dataset, ar.mutualpimode,str(ar.mutual_model_num) + ar.mutual_models[0].replace('.', '_'))
    try:
        os.makedirs(ar.save_dir)
    except:
        pass

    folder_num = len(next(os.walk(ar.save_dir))[1])
    folder_name = osp.join(ar.save_dir, str(folder_num))
    try:
        os.makedirs(folder_name)
    except:
        pass
    ar.save_dir = folder_name

def code_init():
    args = get_arguments()
    pathdir_init(args)
    if args.mode == 'train' and args.submode == 'mutual_kd':
        pass
    else:
        print("Using random seed:{}".format(args.seeds))
        seed_torch(seed=args.seeds)
    if args.submode == 'mutual':
        args.loop = args.mutual_model_num
    sys.stdout = Logger(osp.join(args.save_dir, 'Log.txt'))
    printargs(args)
    return args


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(transf_slices)

def save_checkpoint(saveS,modelS, optimizerS, miouS, epoch, args, i=0):
    for i in range(args.loop):
        model_last_path = os.path.join(args.save_dir, args.name + str(i) + "_last_ckpt.pth")
        model_best_path = os.path.join(args.save_dir, args.name + str(i) + "_BEST_ckpt.pth")
        checkpoint = {
            'epoch': epoch,
            'miou': miouS[i].value()[1],
            'state_dict': modelS[i].state_dict(),
            'optimizer': optimizerS[i].state_dict()
        }

        torch.save(checkpoint, model_last_path)
        # Save arguments
        if saveS[i]:
            torch.save(checkpoint, model_best_path)
            summary_filename = os.path.join(args.save_dir, 'ModelBestInfo_{}.txt'.format(i))
            with open(summary_filename, 'w') as summary_file:
                sorted_args = sorted(vars(args))
                summary_file.write("[Argument Setting]\n")
                if 'kd' in args.submode:
                    unwanted = {'modelname', 'mutual_model_num', 'mutual_models', 'mutualpimode'}
                elif args.submode == 'vanilla':
                    unwanted = {'kdmethod', 'pimode', 'mutual_model_num', 'mutual_models', 'mutualpimode',
                                'teacher_dir',
                                'teacher_model', 'student_model', 'teacher_dir'}
                elif 'mutual' in args.submode:
                    unwanted = {'modelname', 'kdmethod', 'pimode', 'student_model', 'teacher_dir',
                                'teacher_model', }
                new_sorted_args = [e for e in sorted_args if e not in unwanted]
                for arg in new_sorted_args:
                    arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
                    summary_file.write(arg_str)

                summary_file.write("\n[BEST VALIDATION]")
                summary_file.write("\nEpoch: {0}".format(epoch))
                summary_file.write("\nMean IoU: {0}".format(miouS[i].value()[1]))
                summary_file.close()


def load_checkpoint(model, optimizer, args, i=0):
    model_path = os.path.join(args.save_dir, args.name + str(i) + '_BEST_ckpt' + '.pth')
    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou

def train_loss_printer(writer,epoch_lossS,train_IoUS,optimS,epoch,args):
    print("[Epoch: {0:d}] Training".format(epoch))
    if args.submode == 'vanilla':
        for i in range(args.loop):
            if args.tensorboard:
                writer.add_scalar('Train/mIoU', train_IoUS[i].value()[1], epoch)
                writer.add_scalar('Train/ce_loss'.format(i), epoch_lossS[0][i], epoch)
                writer.add_scalar('Learning_rate', optimS[i].param_groups[0]['lr'], epoch)
            trainmsg = "[Model{0:}] [Avg.CE.loss]: {1:.4f}  | Mean IoU: {2:.4f}"
            print(trainmsg.format(i, epoch_lossS[0][i], train_IoUS[i].value()[1]))
    elif args.submode == 'kd':
        for i in range(args.loop):
            if args.tensorboard:
                writer.add_scalar('Train/mIoU', train_IoUS[i].value()[1], epoch)
                writer.add_scalar('Train/ce_loss', epoch_lossS[0][i], epoch)
                writer.add_scalar('Train/{}piloss'.format(args.pimode), epoch_lossS[1][i], epoch)
                writer.add_scalar('Train/paloss', epoch_lossS[2][i], epoch)
                writer.add_scalar('Train/corloss', epoch_lossS[3][i], epoch)
                writer.add_scalar('Learning_rate', optimS[i].param_groups[0]['lr'], epoch)
            trainmsg = "[Model{0:}] [Avg.loss]: {1:.4f} [Avg.Pixel{6:}loss]: {2:.4f} [Avg.Pairloss]: {3:.4f} [Avg.Correlationloss]: {4:.4f} | Mean IoU: {5:.4f}"
            print(trainmsg.format(i, epoch_lossS[0][i], epoch_lossS[1][i], epoch_lossS[2][i], epoch_lossS[3][i],
                                  train_IoUS[i].value()[1], args.pimode))
    elif args.submode == 'mutual':
        for i in range(args.loop):
            if args.tensorboard:
                writer.add_scalar('Train/model{}/mIoU'.format(i), train_IoUS[i].value()[1], epoch)
                writer.add_scalar('Train/model{}/ce_loss'.format(i), epoch_lossS[0][i], epoch)
                writer.add_scalar('Train/model{}/kl_loss'.format(i), epoch_lossS[1][i], epoch)
                writer.add_scalar('Learning_rate/model{}'.format(i), optimS[i].param_groups[0]['lr'], epoch)
            trainmsg = "[Model{0:}] [Avg.CE.loss]: {1:.4f} [Avg.KL.loss]: {2:.4f} | Mean IoU: {3:.4f}"
            print(trainmsg.format(i, epoch_lossS[0][i], epoch_lossS[1][i], train_IoUS[i].value()[1]))

def val_loss_printer(writer,loss,val_IoUS,epoch,args,best_miou):
    isbest = []
    print("[Epoch: {0:d}] Validating".format(epoch))
    for i in range(args.loop):
        if args.tensorboard:
            writer.add_scalar('Validation/model{}/mIoU'.format(i), val_IoUS[i].value()[1], epoch)
            writer.add_scalar('Validation/model{}/ce_loss'.format(i), loss[i], epoch)
        is_best = val_IoUS[i].value()[1] > best_miou[i]
        valmsg = "[Model{0:}] [Avg.CE.loss]: {1:.4f} | Mean IoU: {2:.4f}"
        if is_best:
            valmsg += "[*]"
            best_miou[i] = val_IoUS[i].value()[1]
        print(valmsg.format(i, loss[i], val_IoUS[i].value()[1]))
        isbest.append(is_best)
    return isbest

def printargs(args):
    for key, val in args._get_kwargs():
        print(key + ' : ' + str(val))
