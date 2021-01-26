import os,sys
import torch

from model.model_selector import ModelSelector
import time 
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--model", default="ENet",help=("model architecture"))
parser.add_argument("--width",default = 360, type=int)
parser.add_argument("--height",default = 480,type=int)
parser.add_argument("--dataset",default = 'cityscapes',type=str)
parser.add_argument("--numclass",default = 12, type=int)
parser.add_argument("--ckptdir",type=str)
parser.add_argument("--trained",type=str,default=False)
parser.add_argument("--mode",type=str,default='PC')
args = parser.parse_args()

device = torch.device('cuda')

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.log.write("Python Version:{}.{}\n".format(sys.version_info.major, sys.version_info.minor))
        self.log.write("Torch Version:{}\n".format(torch.__version__))
        self.log.write("Cudnn Version:{}\n".format(torch.backends.cudnn.version()))
        self.log.write("Dataset:{}\n".format(args.dataset))
        self.log.write("Model:{}\n".format(args.model))
        self.log.write("Image width:{}\n".format(args.width))
        self.log.write("Image height:{}\n".format(args.height))
        self.log.write("Num class:{}\n".format(args.numclass))
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
filename = args.model + "_" + args.dataset + "_" + str(args.width) + 'x' + str(args.height) + '_' + args.mode +  '.txt'
sys.stdout = Logger(filename)


def load_checkpoint(model,ckpt_dir):
	checkpoint = torch.load(ckpt_dir)
	model.load_state_dict(checkpoint['state_dict'])
	return model


def test(model,loop):
	allfps = 0
	for i in range(loop):
		fake_tensor = torch.rand(1,3,args.width,args.height).to(device)
		tic = time.time()
		outputs = model(fake_tensor)
		toc = time.time()
		tttime = toc-tic
		fps = 1/ tttime
		print('loop{}:{}'.format(i,fps))
		allfps = allfps + fps
	fps_avg = allfps / loop
	print('avg_fps:{}'.format(fps_avg))
selector = ModelSelector(args.numclass)
model = selector.segmentor(args.model).to(device)
paramsg = '[*] Number of parameters of {} model: {:,}'
print(paramsg.format(args.model,sum([p.data.nelement() for p in model.parameters()])))
test(model,5)
