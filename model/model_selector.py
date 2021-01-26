from model.net.enet import ENet
from model.config.enet_config import ENET_MODEL_DICT
from model.deeplab import *
# from ptflops import get_model_complexity_info


class ModelSelector(object):
    def __init__(self, numclass):
        self.numclass = numclass

    def segmentor(self, model_name):
        if 'ENet' in model_name:
            cs = ENET_MODEL_DICT[model_name]['channel_set']
            r2s = ENET_MODEL_DICT[model_name]['reduce2stage']
            r3s = ENET_MODEL_DICT[model_name]['reduce3stage']
            stg3 = ENET_MODEL_DICT[model_name]['stage3']
            model = ENet(num_classes=self.numclass, CHset=cs, reduce2stage=r2s, reduce3stage=r3s, stage3=stg3)
        elif 'Deeplab' in model_name:
            model = DeepLab(num_classes=self.numclass, backbone='mobilenet', output_stride=8, sync_bn=False,
                            freeze_bn=False)
        else:
            raise RuntimeError("Argument model {} is not defined !".format(model_name))
        return model


if __name__ == '__main__':
    selector = ModelSelector(19)
    # model = selector.segmentor("ENet_3enc0_channel0.5")
    # model = selector.segmentor("ENet_3enc0")
    # weight =torch.load('/media/tan/Disk/graduate_Code/ENet_ncist/pre-trained/ENet_rd3st_c_512x512_mfb.pth')
    # weight =weight['state_dict']
    # model.load_state_dict(weight)
    for model_name in ENET_MODEL_DICT:
        # print(model_name)
        model = selector.segmentor(model_name)
        # print(id(model))
        paramsg = '[*] Number of parameters of {} model: {:,}'
        print(paramsg.format(model_name, sum([p.data.nelement() for p in model.parameters()])))
