import  os

import numpy as np
from PIL import Image
import natsort

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision
from torchvision import datasets, models, transforms
from utils.preprocess import transpose, normalize, TransformTwice
from torch.utils.data.sampler import SubsetRandomSampler


INPUT_DIR_PATH = r"./data"
DOMAIN = "domain_hawkice_damdev"
DOMAIN_LIST = ["domain_hawkice_damdev", "domain_hawkice_longphong", "domain_inhouse_public"]


def get_gender_data_from_all_domains(root, domain_list, 
                 transform_train=None, transform_val=None,
                 val_split = 0.2,
                 download=True):
    
    # Train & Val
    train_dataset_list = []
    train_dataset = {}
    for domain in domain_list:
        train_labeled_path = os.path.join(root, domain, 'train')
        train_labeled_set = datasets.ImageFolder(train_labeled_path, transform=transform_train)
        train_dataset[domain] = Data_labeled(train_labeled_set, 
                                train=True, transform=transform_train)
        train_dataset_list.append(train_dataset[domain])

    train_dataset_all = ConcatDataset(train_dataset_list)
    train_datasets  = dict()
    train_datasets['train'], \
        train_datasets['val'] = random_split(train_dataset_all, 
                                            (round( (1-val_split)*len(train_dataset_all) ), 
                                            round( val_split*len(train_dataset_all) ) )
                                            )

    # Test
    test_dataset_list = []
    test_dataset = {}
    for domain in domain_list:
        test_path = os.path.join(root, domain, 'test')
        test_set = datasets.ImageFolder(test_path, transform=transform_val)
        test_dataset[domain] = Data_labeled(test_set, train=False, 
                                   transform=transform_val)
        test_dataset_list.append(test_dataset[domain])
    test_dataset_all = ConcatDataset(test_dataset_list)

    # Unlabel
    train_unlabeled_path = os.path.join(root, 'unlabeled')
    train_unlabeled_dataset = Data_unlabeled(train_unlabeled_path,  
                                                train=True, 
                                                transform=TransformTwice(transform_train))

    print (f"#Labeled: {len(train_datasets['train'])} \
           #Unlabeled: {len(train_unlabeled_dataset)} \
           #Val: {len(train_datasets['val'])}")
    return train_datasets, train_unlabeled_dataset, test_dataset



def get_gender_data(root, domain, n_labeled, n_class,
                 transform_train=None, transform_val=None,
                 val_split = 0.2,
                 download=True):
    train_labeled_path = os.path.join(root, domain, 'train')
    train_labeled_set = datasets.ImageFolder(train_labeled_path, transform=transform_train)

    test_path = os.path.join(root, domain, 'test')
    test_set = datasets.ImageFolder(test_path, transform=transform_val)

    # Train & Val 
    train_labeled_idxs, val_idxs = train_val_split(train_labeled_set.targets, 
                                                    n_labeled, n_class,
                                                    val_split)
    train_sampler = SubsetRandomSampler(train_labeled_idxs)
    valid_sampler = SubsetRandomSampler(val_idxs)
    train_dataset = Data_labeled(train_labeled_set, 
                                train=True, transform=transform_train)
   
    # Test 
    test_dataset = Data_labeled(test_set, train=False, 
                                   transform=transform_val)
    
    # Unlabel
    train_unlabeled_path = os.path.join(root, 'unlabeled')
    train_unlabeled_dataset = Data_unlabeled(train_unlabeled_path,  
                                                train=True, 
                                                transform=TransformTwice(transform_train))

    print (f"#Labeled: {len(train_labeled_idxs)} \
           #Unlabeled: {len(train_unlabeled_dataset)} \
           #Val: {len(val_idxs)}")
    return train_dataset, train_sampler, valid_sampler, train_unlabeled_dataset, test_dataset



def train_val_split(labels, n_labeled, n_class, val_split):
    labels = np.array(labels)
    train_labeled_idxs = []
    val_idxs = []

    for i in range(n_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        split = int(np.floor(val_split * len(idxs)))
        train_labeled_idxs.extend(idxs[:-split])
        val_idxs.extend(idxs[-split:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, val_idxs


class Data_labeled(Dataset):
    def __init__(self, root, 
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Data_labeled, self).__init__()
        self.data = root
        self.targets = root.targets
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
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, target


class Data_unlabeled(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.image_paths = root
        self.transform = transform
        self.all_imgs = os.listdir(root)
        # self.total_imgs = len(self.all_imgs)
        self.total_imgs = natsort.natsorted(self.all_imgs)
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

