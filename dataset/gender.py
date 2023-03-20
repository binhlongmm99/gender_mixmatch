import  os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.preprocess import transpose, normalize, TransformTwice
from torch.utils.data.sampler import SubsetRandomSampler


INPUT_DIR_PATH = r"./data"
DOMAIN = "domain_hawkice_damdev"


def get_gender_data(root, domain, n_labeled, n_class,
                 transform_train=None, transform_val=None,
                 download=True):
    train_labeled_path = os.path.join(root, domain, 'train')
    dataset_labeled_train = datasets.ImageFolder(train_labeled_path, transform=transform_train)

    test_path = os.path.join(root, domain, 'test')
    dataset_test = datasets.ImageFolder(test_path, transform=transform_val)

    # Train & Val 
    # train_labeled_idxs, train_unlabeled_idxs, \
    #         val_idxs = train_val_split(dataset_labeled_train.targets, n_labeled, n_class)
    train_labeled_idxs, val_idxs = train_val_split(dataset_labeled_train.targets, 
                                                   n_labeled, n_class,
                                                   val_split = 0.2
                                                   )
    # train_labeled_dataset = Data_labeled(dataset_labeled_train, train_labeled_idxs, 
    #                                         train=True, transform=transform_train)
    # val_dataset = Data_labeled(dataset_labeled_train, val_idxs, train=True, 
    #                               transform=transform_val)
    
    train_sampler = SubsetRandomSampler(train_labeled_idxs)
    valid_sampler = SubsetRandomSampler(val_idxs)
    train_dataset = Data_labeled(dataset_labeled_train, 
                                train=True, transform=transform_train)
   
    # Test 
    test_dataset = Data_labeled(dataset_test, train=False, 
                                   transform=transform_val)
    
    # Unlabel
    train_unlabeled_path = os.path.join(root, 'unlabeled')
    # dataset_unlabeled_train = datasets.ImageFolder(train_unlabeled_path, transform=transform_train) 
    train_unlabeled_dataset = Data_unlabeled(train_unlabeled_path,  
                                                train=True, 
                                                transform=TransformTwice(transform_train))

    print (f"#Labeled: {len(train_labeled_idxs)} \
           #Unlabeled: {len(train_unlabeled_dataset)} \
           #Val: {len(val_idxs)}")
    # return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    return train_dataset, train_sampler, valid_sampler, train_unlabeled_dataset, test_dataset

def train_val_split(labels, n_labeled, n_class, val_split):
    labels = np.array(labels)
    # print(len(labels))
    # print()
    train_labeled_idxs = []
    val_idxs = []

    for i in range(n_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        split = int(np.floor(val_split * len(idxs)))
        # print(split)
        train_labeled_idxs.extend(idxs[:-split])
        val_idxs.extend(idxs[-split:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, val_idxs


class Data_labeled(Dataset):
    def __init__(self, root, 
                #  indexs=None, 
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Data_labeled, self).__init__()
        # self.image_paths = root
        self.data = root
        self.targets = root.targets
        # if indexs is not None:
        #     self.data = np.array(root)[indexs]
        #     self.targets = np.array(root.targets)[indexs]
        # self.data = transpose(normalize(self.data))

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index]
        # img, target = self.data[index], self.targets[index]
        # print(type(img))
        # print(type(target))
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, target


class Data_unlabeled(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = root
        # self.data = root
        self.transform = transform
        self.all_imgs = os.listdir(root)
        self.total_imgs = len(self.all_imgs)
        self.targets = np.array([-1 for _ in range(len(self.all_imgs))])

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.image_paths, self.all_imgs[index])
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            img = self.transform(image)
        target = self.targets[index]
        return img, target


# class Data_unlabeled(Dataset):
#     def __init__(self, root, train=True, transform=None):
#         self.image_paths = root
#         # self.data = root
#         self.transform = transform
#         self.targets = np.array([-1 for _ in range(len(self.image_paths))])
#         self.all_imgs = os.listdir(root)
#         # self.total_imgs = natsorted(all_imgs)

#     def __len__(self):
#         return len(self.all_imgs)

#     def __getitem__(self, idx):
#         img_loc = os.path.join(self.image_paths, self.all_imgs[idx])
#         image = Image.open(img_loc).convert("RGB")
#         if self.transform is not None:
#             img = self.transform(image)
#         target = -1
#         return img, target

#     # def __len__(self):
#     #     return len(self.image_paths)
            
#     # def __getitem__(self, index):
#     #     image_path = self.image_paths[index]
#     #     img = Image.open(image_path)
#     #     # y = self.get_class_label(image_path.split('/')[-1])
#     #     # img, target = self.data[index], self.targets[index]
#     #     # img, target = self.image_paths[index]
#     #     if self.transform is not None:
#     #         img = self.transform(img)
#     #     target = -1
#     #     return img, target
    
#     # def get_class_label(self, image_name):
#     #     # your method here
#     #     return -1


# class Data_unlabeled(Dataset):
#     def __init__(self, root, indexs, train=True,
#                  transform=None, target_transform=None,
#                  download=False):
#         super(Data_unlabeled, self).__init__(root, indexs, train=train,
#                  transform=transform, target_transform=target_transform,
#                  download=download)
#         self.targets = np.array([-1 for i in range(len(self.targets))])

