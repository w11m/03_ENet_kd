import os
from collections import OrderedDict
import torch.utils.data as data
from PIL import Image
from . import data_utils
import copy

class Cityscapes(data.Dataset):

    """Cityscapes dataset https://www.cityscapes-dataset.com/.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = "leftImg8bit/train"
    train_lbl_folder = "gtFine/train"

    # Validation dataset root folders
    val_folder = "leftImg8bit/val"
    val_lbl_folder = "gtFine/val"

    # Test dataset root folders
    test_folder = "leftImg8bit/test"
    test_lbl_folder = "gtFine/test"

    # Demo Video root folders
    demo_folder = test_folder

    # Filters to find the images
    img_extension = '.png'
    lbl_name_filter = 'labelIds'

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, -1)
    # The values above are remapped to the following
    new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                   8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('road', (128, 64, 128)),
        ('sidewalk', (244, 35, 232)),
        ('building', (70, 70, 70)),
        ('wall', (102, 102, 156)),
        ('fence', (190, 153, 153)),
        ('pole', (153, 153, 153)),
        ('traffic_light', (250, 170, 30)),
        ('traffic_sign', (220, 220, 0)),
        ('vegetation', (107, 142, 35)),
        ('terrain', (152, 251, 152)),
        ('sky', (70, 130, 180)),
        ('person', (220, 20, 60)),
        ('rider', (255, 0, 0)),
        ('car', (0, 0, 142)),
        ('truck', (0, 0, 70)),
        ('bus', (0, 60, 100)),
        ('train', (0, 80, 100)),
        ('motorcycle', (0, 0, 230)),
        ('bicycle', (119, 11, 32))
    ])
    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=data_utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = data_utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = data_utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = data_utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = data_utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = data_utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = data_utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                name_filter=self.lbl_name_filter,
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'submit':
            # Get the demo video
            self.demo_data = data_utils.get_files(
                os.path.join(root_dir, self.demo_folder),
                extension_filter=self.img_extension)
            self.demo_imgpath = data_utils.get_path(
                os.path.join(root_dir, self.demo_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train' or self.mode.lower() == 'train_kd':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        elif self.mode.lower() == 'submit':
            data_path, img_name = self.demo_data[index], self.demo_imgpath[index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
        if self.mode.lower() == 'submit':
            img = Image.open(data_path)
        else:
            img, label = self.loader(data_path, label_path)

        if self.mode.lower() == 'submit':
            pass
        else:
            # Remap class labels
            label = data_utils.remap(label, self.full_classes, self.new_classes)

        if self.transform is not None and self.mode == 'train_kd':
            img_high = img
            img_norm = copy.copy(img)
            img_high = self.transform[0](img_high)
            img_norm = self.transform[1](img_norm)
        elif self.transform is not None:
            img = self.transform(img)

        if self.mode.lower() == 'submit':
            pass
        elif self.mode.lower() == 'train_kd':
            if self.label_transform is not None:
                label_high = label
                label_norm = copy.copy(label)
                label_high = self.label_transform[0](label_high)
                label_norm = self.label_transform[1](label_norm)
        else:
            if self.label_transform is not None:
                label = self.label_transform(label)
        if self.mode.lower() == 'submit':
            return img, img_name
        elif self.mode.lower() == 'train_kd':
            return img_high, img_norm, label_high, label_norm
        else:
            return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train' or self.mode.lower() == 'train_kd':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        elif self.mode.lower() == 'submit':
            return len(self.demo_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")