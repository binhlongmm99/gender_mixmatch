import  os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from preprocess import transpose, normalize, TransformTwice

INPUT_DIR_PATH = r"E:\icomm\data"
DOMAIN = "domain_hawkice_damdev"


def get_gender_data(root, domain, n_labeled, n_class,
                 transform_train=None, transform_val=None,
                 download=True):
    train_labeled_path = os.path.join(root, domain, 'train')
    dataset_labeled_train = datasets.ImageFolder(train_labeled_path, transform=transform_train)
    train_unlabeled_path = os.path.join(root, 'unlabeled')
    dataset_unlabeled_train = datasets.ImageFolder(train_unlabeled_path, transform=transform_train)

    test_path = os.path.join(root, domain, 'test')
    dataset_test = datasets.ImageFolder(test_path, transform=transform_train)

    # Train & Val 
    # train_labeled_idxs, train_unlabeled_idxs, \
    #         val_idxs = train_val_split(dataset_labeled_train.targets, n_labeled, n_class)
    train_labeled_idxs, val_idxs = train_val_split(dataset_labeled_train.targets, 
                                                   n_labeled, n_class,
                                                   val_split = 0.2
                                                   )
    train_labeled_dataset = Data_labeled(dataset_labeled_train, train_labeled_idxs, 
                                            train=True, transform=transform_train)
    val_dataset = Data_labeled(dataset_labeled_train, val_idxs, train=True, 
                                  transform=transform_val)
    # Unlabel 
    train_unlabeled_dataset = Data_unlabeled(dataset_unlabeled_train,  
                                                train=True, transform=TransformTwice(transform_train))
    
    # Test 
    test_dataset = Data_labeled(dataset_test, train=False, 
                                   transform=transform_val)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(dataset_unlabeled_train)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(labels, n_labeled, n_class, val_split):
    # dataset = CustomDatasetFromCSV(my_path)
    # batch_size = 16
    # validation_split = .2
    # shuffle_dataset = True
    # random_seed= 42

    # # Creating data indices for training and validation splits:
    # dataset_size = len(dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]

    labels = np.array(labels)
    train_labeled_idxs = []
    val_idxs = []

    for i in range(n_class):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        split = int(np.floor(val_split * len(idxs)))
        train_labeled_idxs.extend(idxs[:split])
        val_idxs.extend(idxs[-split:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, val_idxs



class Data_labeled(Dataset):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Data_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class Data_unlabeled(Dataset):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(Data_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])

