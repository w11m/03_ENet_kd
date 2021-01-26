ENET_MODEL_DICT = {
    'ENet': {
        'channel_set':[16,64,128],
        'reduce2stage':False,
        'reduce3stage':False,
        'stage3':True,
    },
    'ENet_slim0.75': {
        'channel_set': [16, 48, 96],
        'reduce2stage': False,
        'reduce3stage': False,
        'stage3': True,
    },
    'ENet_slim0.5': {
        'channel_set': [16, 32, 64],
        'reduce2stage': False,
        'reduce3stage': False,
        'stage3': True,
    },
    'ENet_slim0.25': {
        'channel_set': [8, 16, 32],
        'reduce2stage': False,
        'reduce3stage': False,
        'stage3': True,
    },
    'ENet_2enc0.5_3enc0.5': {
        'channel_set': [16, 64, 128],
        'reduce2stage': True,
        'reduce3stage': True,
        'stage3': True,
    },
    'ENet_2enc0.5': {
        'channel_set': [16, 64, 128],
        'reduce2stage': True,
        'reduce3stage': False,
        'stage3': True,
    },
    'ENet_3enc0': {
        'channel_set': [16, 64, 128],
        'reduce2stage': False,
        'reduce3stage': True,
        'stage3': False,
    },
    'ENet_3enc0_channel0.75': {
        'channel_set': [16, 48, 96],
        'reduce2stage': False,
        'reduce3stage': True,
        'stage3': False,
    },
    'ENet_3enc0_channel0.6': {
        'channel_set': [16, 48, 80],
        'reduce2stage': False,
        'reduce3stage': True,
        'stage3': False,
    },
}
