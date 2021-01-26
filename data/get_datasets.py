from .camvid import CamVid
from .cityscapes import Cityscapes
import mytransforms
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image

import torch
import os, sys
from . import data_utils
import numpy as np


def select_dataset(set_name='CamVid'):
    if set_name == 'CamVid':
        return CamVid
    elif set_name == 'Cityscapes':
        return Cityscapes


def load_npy(npy_path):
    with open(npy_path, 'r'):
        try:
            data = np.load(npy_path).item()
        except:
            data = np.load(npy_path)
    return data


def save_npy(npy_path, data):
    with open(npy_path, 'w'):
        np.save(npy_path, data)


def load_dataset(args, dataset):
    print("\nLoading dataset...\n")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_train_transform = transforms.Compose([transforms.Resize((args.height, args.width)), transforms.ToTensor()])
    label_train_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width), Image.NEAREST), mytransforms.PILToLongTensor()])
    image_valtest_transform = transforms.Compose([transforms.Resize((args.height, args.width)), transforms.ToTensor()])
    label_valtest_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width), Image.NEAREST), mytransforms.PILToLongTensor()])
    # try:
    #     if args.multiRES:
    #         img_trans = [image_high_transform, image_norm_transform]
    #         label_trans = [label_high_transform, label_norm_transform]
    # except:
    #     pass

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(args.dataset_dir, mode='train', transform=image_train_transform,
                        label_transform=label_train_transform)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(args.dataset_dir, mode='val', transform=image_valtest_transform,
                      label_transform=label_valtest_transform)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # # Load the test set as tensors
    if args.dataset.lower() == 'cityscapes':
        submit_set = dataset(args.dataset_dir, mode='submit', transform=image_valtest_transform,
                             label_transform=label_valtest_transform)
        test_loader = data.DataLoader(submit_set, batch_size=2, shuffle=False, num_workers=args.workers)
    elif args.dataset.lower() == 'camvid':
        test_set = dataset(args.dataset_dir, mode='test', transform=image_valtest_transform,
                           label_transform=label_valtest_transform)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)
    args.numclass = num_classes
    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    images, labels = iter(train_loader).next()
    # print("Image high size:", images_high.size())
    # print("Label high size:", labels_high.size())
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")

    weighting_name = "{}_{}x{}_{}.npy".format(args.dataset,args.width,args.height,args.weighing.lower())
    try:
        os.mkdir(args.weighing_dir)
    except:
        pass
    weighing_dir = os.path.join(args.weighing_dir, weighting_name)
    print(weighing_dir)
    class_weights = 0
    if args.weighing.lower() == 'enet':
        if os.path.exists(weighing_dir):
            class_weights = load_npy(weighing_dir)
        else:
            class_weights = data_utils.enet_weighing(train_loader, num_classes)
            save_npy(weighing_dir, class_weights)
    elif args.weighing.lower() == 'mfb':
        if os.path.exists(weighing_dir):
            class_weights = load_npy(weighing_dir)
        else:
            class_weights = data_utils.median_freq_balancing(train_loader, num_classes)
            save_npy(weighing_dir, class_weights)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to('cuda')
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, class_encoding


def load_submit_dataset(args, dataset):
    print("\nLoading dataset...\n")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)
    # high resolution transformation

    image_norm_transform = transforms.Compose([transforms.Resize((args.height, args.width)), transforms.ToTensor()])
    label_norm_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width), Image.NEAREST), mytransforms.PILToLongTensor()])
    # try:
    #     if args.multiRES:
    #         img_trans = [image_high_transform, image_norm_transform]
    #         label_trans = [label_high_transform, label_norm_transform]
    # except:
    #     pass
    img_trans = image_norm_transform
    label_trans = label_norm_transform
    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(args.dataset_dir, mode='train', transform=img_trans, label_transform=label_trans)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # # Load the test set as tensors
    submit_set = dataset(args.dataset_dir, mode='submit', transform=image_norm_transform,
                         label_transform=label_norm_transform)
    submit_loader = data.DataLoader(submit_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)
    args.numclass = num_classes
    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Test dataset size:", len(train_set))
    print("Class-color encoding:", class_encoding)

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")

    weighting_name = "{}_{}x{}_{}.npy".format(args.dataset, args.width, args.height, args.weighing.lower())
    try:
        os.mkdir(args.weighing_dir)
    except:
        pass
    weighing_dir = os.path.join(args.weighing_dir, weighting_name)
    print(weighing_dir)
    class_weights = 0
    if args.weighing.lower() == 'enet':
        if os.path.exists(weighing_dir):
            class_weights = load_npy(weighing_dir)
        else:
            class_weights = data_utils.enet_weighing(train_loader, num_classes)
            save_npy(weighing_dir, class_weights)
    elif args.weighing.lower() == 'mfb':
        if os.path.exists(weighing_dir):
            class_weights = load_npy(weighing_dir)
        else:
            class_weights = data_utils.median_freq_balancing(train_loader, num_classes)
            save_npy(weighing_dir, class_weights)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to('cuda')
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, submit_loader), class_weights, class_encoding
