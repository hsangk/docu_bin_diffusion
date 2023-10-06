import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader, random_split
import bisect
from os import listdir

# taming/data/custom.py
class CustomBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        # path2image = './data/'
        # self.path2data = os.path.join(path2image, 'bitrain')
        # self.path2gt = os.path.join(path2image,'bitrain_gt')
        # self.data_filenames = [x for x in listdir(self.path2data)]
        # self.gt_filenames = [x for x in listdir(self.path2gt)]
        self.process_images = True
        self._load()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()
            l1 = len(self.relpaths)

        with open(self.txt_gtlist, "r") as f:
            self.gtpaths = f.read().splitlines()
            l2 = len(self.gtpaths)

        labels = {
            "relpath": np.array(self.relpaths),
            "gtpath" : np.array(self.gtpaths)
        }

        if self.process_images:
            self.data = ImagePaths(self.relpaths,
                                   labels=labels,
                                   size=self.size,
                                   )
        else:
            self.data = self.relpaths


# dataset = CustomBase()



class CustomTrain(CustomBase):
    def __init__(self, size, path2image= './data/', **kwargs):
    # def __init__(self, size, training_images_list_file='./data/train.txt'):
        super().__init__(**kwargs)
        self.path2image = './data/'
        self.path2data = os.path.join(path2image, 'bitrain')
        self.path2gt = os.path.join(path2image,'bitrain_gt')
        self.data_filenames = [x for x in listdir(self.path2data)]
        self.gt_filenames = [x for x in listdir(self.path2gt)]
        self.process_images = True
        self._load()

    def _load(self):
        l1 = len(self.data_filenames)
        l2 = len(self.gt_filenames)

        labels = {
            "relpath": np.array(self.data_filenames),
            "label": np.array(range(len(self.data_filenames)))
        }

        if self.process_images:
            self.data = ImagePaths(self.data_filenames,
                                   labels=labels,
                                   size=self.size,
                                   )
        else:
            self.data = self.relpaths




# data_dir = './data/train.txt'
# a = CustomTrain(256, data_dir)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file = './data/test.txt'):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)




# taming/data/base.py

# class ConcatDatasetWithIndex(ConcatDataset):
#     """Modified from original pytorch code to return dataset idx"""
#     def __getitem__(self, idx):
#         if idx < 0:
#             if -idx > len(self):
#                 raise ValueError("absolute value of index should not exceed dataset length")
#             idx = len(self) + idx
#         dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
#         if dataset_idx == 0:
#             sample_idx = idx
#         else:
#             sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
#         return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):

        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
