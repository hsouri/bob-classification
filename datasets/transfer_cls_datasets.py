"""
Datasets for transfer learning
Author: Manli Shu
"""

import os
import json
import scipy.io as sio
import random
import csv
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split

from torchvision.transforms.transforms import CenterCrop
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import urllib

transfer_datasets = {
    'flower102': 'Flower102',
    'aircraft': 'Aircraft',
    # 'birdsnap': 'Birdsnap',
    'dtd': 'DTD',
    'voc2007': 'VOC2007',
    'pets': 'Pets',
    'sun397': 'SUN397',
    'cars': 'Cars',
    'food101': 'Food101',
    'caltech101': 'Caltech101',
    'cifar10': 'Cifar10',
    'cifar100': 'Cifar100',
}

class Flower102(Dataset):
    """ Oxford Flower102 dataset """
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        set_ids = sio.loadmat(os.path.join(root, 'setid.mat'))
        labels = sio.loadmat(os.path.join(root, 'imagelabels.mat'))
        labels = labels['labels'][0]
        if mode == 'train':
            self.ids = set_ids['trnid'][0]
        elif mode in ['val', 'validation']:
            self.ids = set_ids['valid'][0]
        elif mode == 'trainval':
            self.ids = np.concatenate((set_ids['trnid'][0], set_ids['valid'][0]), axis=0)
        elif mode == 'test':
            self.ids = set_ids['tstid'][0]
        else:
            raise NotImplementedError
        self.image_list = ['image_'+'{:s}.jpg'.format(str(i).zfill(5)) for i in self.ids]
        self.label_list = [labels[i-1]-1 for i in self.ids]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'jpg', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'data/aircraft_list', '{:s}.txt'.format('test' if self.mode=='test' else 'train')), 'r') as fp:
            if self.mode == 'train':
                lines = [s.replace("\n", "") for s in fp.readlines()[:3334]]
            elif self.mode == 'val':
                lines = [s.replace("\n", "") for s in fp.readlines()[3334:]]
            elif self.mode in ['trainval', 'test']:
                lines = [s.replace("\n", "") for s in fp.readlines()]
            else:
                raise NotImplementedError
        for line in lines:
            self.image_list.append(line.split(" ")[0])
            self.label_list.append(int(line.split(" ")[1])-1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'data', 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class Birdsnap(Dataset):
    """ Birdsnap dataset """
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        self.image_list = []
        self.label_list = []
        if not self.mode == 'test':
            with open(os.path.join(self.path, 'images.txt'), 'r') as fp:
                reader = csv.DictReader(fp, delimiter='\t')
                for line in reader:
                    self.image_list.append(line['path'])
                    self.label_list.append(int(line['species_id'])-1)
            if self.mode in ['train', 'val']:
                train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
                train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
                self.image_list = train_image if self.mode == 'train' else val_image
                self.label_list = train_label if self.mode == 'train' else val_label
        else:
            with open(os.path.join(self.path, 'test_images.txt'), 'r') as fp:
                self.image_list = [s.replace("\n", "") for s in fp.readlines()[1:]]
            mapping_dict = {}
            with open(os.path.join(self.path, 'species.txt'), 'r') as fp:
                reader = csv.DictReader(fp, delimiter='\t')
                for row in reader:
                    mapping_dict[row['dir']] = int(row['id'])-1
            for path in self.image_list:
                dir = path.split("/")[0]
                self.label_list.append(mapping_dict[dir.lower()])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class DTD(Dataset):
    """ Describable Texture dataset """
    def __init__(self, root, mode='train', split=1, transform=None):
        self.transform = transform
        self.split = split
        self.path = root
        self.mode = mode
        self.image_list = []
        self.label_list = []
        if self.mode == 'trainval':
            with open(os.path.join(self.path, 'labels', 'train{:d}.txt'.format(self.split)), 'r') as fp:
                self.image_list = [s.replace("\n", "") for s in fp.readlines()]
            with open(os.path.join(self.path, 'labels', 'val{:d}.txt'.format(self.split)), 'r') as fp:
                self.image_list.extend([s.replace("\n", "") for s in fp.readlines()])
        else:
            with open(os.path.join(self.path, 'labels', '{:s}{:d}.txt'.format(self.mode, self.split)), 'r') as fp:
                self.image_list = [s.replace("\n", "") for s in fp.readlines()]
        meta = sio.loadmat(os.path.join(self.path, 'imdb', 'imdb.mat'))['meta']
        meta = list(meta[0][0][1][0])
        for path in self.image_list:
            dir = path.split("/")[0]
            self.label_list.append(meta.index(dir))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class VOC2007(torchvision.datasets.VOCDetection):
    def __init__(self, root, mode='train', year='2007', transform=None):
        super(VOC2007, self).__init__(root, year='2007', image_set=mode, transform=transform)
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        labels = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())    
        name = labels['annotation']['object'][0]['name']
        assert name in self.classes, "Unknown class: {:s}".format(name)
        target = self.classes.index(name)
        if self.transform is not None:
            img = self.transform(img)        
        return img, torch.tensor(target).long()


class Pets(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super().__init__()
        self.root = os.path.join(root, 'images')
        self.mode = mode
        self.transform = transform
        self.image_list = [fname for fname in os.listdir(self.root) if fname.endswith('.jpg')]
        self.classes = list(set(''.join(fname.split('_')[:-1]) for fname in self.image_list))
        self.label_list = [self.classes.index(''.join(p.split('_')[:-1])) for p in self.image_list]

        trainval_image, test_image = train_test_split(np.array(self.image_list), test_size=0.15, random_state=0)
        trainval_label, test_label = train_test_split(np.array(self.label_list), test_size=0.15, random_state=0)
        if self.mode in ['test', 'trainval']:
            self.image_list = test_image if self.mode == 'test' else trainval_image
            self.label_list = test_label if self.mode == 'test' else trainval_label
        elif self.mode in ['train', 'val']:
            train_image, val_image = train_test_split(np.array(trainval_image), test_size=0.1, random_state=0)
            train_label, val_label = train_test_split(np.array(trainval_label), test_size=0.1, random_state=0)
            self.image_list = train_image if self.mode == 'train' else val_image
            self.label_list = train_label if self.mode == 'train' else val_label
        else:
            raise NotImplementedError("Unknown data split {:s}".format(self.mode))
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

class Caltech101(torchvision.datasets.Caltech101):
    def __init__(self, root, mode='train', transform=None):
        super(Caltech101, self).__init__(root, transform=transform, download=True)
        self.mode = mode
        self.label_list = self.y
        self.image_list = [os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[idx]],
                                      "image_{:04d}.jpg".format(self.index[idx])) for idx in self.index]
        trainval_image, test_image = train_test_split(np.array(self.image_list), test_size=0.4, random_state=0)
        trainval_label, test_label = train_test_split(np.array(self.label_list), test_size=0.4, random_state=0)
        if self.mode in ['test', 'trainval']:
            self.image_list = test_image if self.mode == 'test' else trainval_image
            self.label_list = test_label if self.mode == 'test' else trainval_label
        elif self.mode in ['train', 'val']:
            train_image, val_image = train_test_split(np.array(trainval_image), test_size=0.1, random_state=0)
            train_label, val_label = train_test_split(np.array(trainval_label), test_size=0.1, random_state=0)
            self.image_list = train_image if self.mode == 'train' else val_image
            self.label_list = train_label if self.mode == 'train' else val_label
        else:
            raise NotImplementedError("Unknown data split {:s}".format(self.mode))
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class SUN397(Dataset):
    """ Describable Texture dataset """
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.class_names = []
        with open(os.path.join(self.path, 'ClassName.txt'), 'r') as fp:
            self.class_names = [s.replace("\n", "").split("/")[-1] for s in fp.readlines()]

        if self.mode == 'test':
            fp = open(os.path.join(self.path, 'Testing_01.txt'), 'r')
        else:
            fp = open(os.path.join(self.path, 'Training_01.txt'), 'r')

        for s in fp.readlines():
            s = s.replace("\n", "")
            label_name = s.split("/")[-2]
            self.image_list.append(s[1:]) # discard the first '/' for os.path.join to work
            self.label_list.append(self.class_names.index(label_name))
        fp.close()
        if self.mode in ['train', 'val']:
            train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
            train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
            self.image_list = train_image if self.mode == 'train' else val_image
            self.label_list = train_label if self.mode == 'train' else val_label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'SUN397', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class Cars(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        meta = sio.loadmat(os.path.join(self.path, \
            'devkit/cars_{:s}_annos.mat'.format('test' if self.mode == 'test' else 'train')))['annotations']

        self.image_list = [x[-1][0] for x in meta[0]]
        self.label_list = [int(x[-2][0][0])-1 for x in meta[0]]
        # if self.mode in ['train', 'val']:
        #     train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
        #     train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
        #     self.image_list = train_image if self.mode == 'train' else val_image
        #     self.label_list = train_label if self.mode == 'train' else val_label
        # else:
        #     raise NotImplementedError

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'cars_{:s}'.format(\
            'test' if self.mode == 'test' else 'train'), self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class Food101(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.path = os.path.join(root, "food-101")
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.class_names = []
        with open(os.path.join(self.path, 'meta/classes.txt'), 'r') as fp:
            self.class_names = [s.replace("\n", "") for s in fp.readlines()]

        with open(os.path.join(self.path, 'meta', '{:s}.txt'.format(\
            'test' if self.mode == 'test' else 'train')), 'r') as fp:
            for s in fp.readlines():
                s = s.replace("\n", "")
                label_name = s.split("/")[0]
                self.image_list.append('{:s}.jpg'.format(s))
                self.label_list.append(self.class_names.index(label_name))
        # if self.mode in ['train', 'val']:
        #     train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
        #     train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
        #     self.image_list = train_image if self.mode == 'train' else val_image
        #     self.label_list = train_label if self.mode == 'train' else val_label
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

class Cifar100(torchvision.datasets.CIFAR100):
    def __init__(self, root, mode='train', transform=None):
        super(Cifar100, self).__init__(root, train=True if mode=='train' else False, \
                                    transform=transform, download=True)
        self.mode = mode
        self.label_list = self.targets
        self.image_list = self.data

        # if self.mode in ['train', 'val']:
        #     train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
        #     train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
        #     self.image_list = train_image if self.mode == 'train' else val_image
        #     self.label_list = train_label if self.mode == 'train' else val_label
        # elif self.mode == 'train_all'

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_list[idx])
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root, mode='train', transform=None):
        super(Cifar10, self).__init__(root, train=True if mode=='train' else False, \
                                    transform=transform, download=True)
        self.mode = mode
        self.label_list = self.targets
        self.image_list = self.data

        # if self.mode in ['train', 'val']:
        #     train_image, val_image = train_test_split(np.array(self.image_list), test_size=0.1, random_state=0)
        #     train_label, val_label = train_test_split(np.array(self.label_list), test_size=0.1, random_state=0)
        #     self.image_list = train_image if self.mode == 'train' else val_image
        #     self.label_list = train_label if self.mode == 'train' else val_label
        # elif self.mode == 'train_all'

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_list[idx])
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()

class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            c_range = max(self.label_list) + 1
            if isinstance(n_shot, float):
                # sample a percentage of training data
                n_shot = int((len(self.image_list) * n_shot) / c_range)

            few_shot_samples = []
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


class SemiImageNet(Dataset):
    def __init__(self, image_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.image_list = []
        self.label_list = []
        train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{int(n_shot)}percent.txt').readlines()
        is_train = (mode == 'train')
        root = os.path.join(image_path, 'train' if is_train else 'val')
        dataset= datasets.ImageFolder(root)

        if is_train:
            for fname in train_files:
                fname = fname.decode().strip()
                cls = fname.split('_')[0]

                self.image_list.append(self.image_path + '/train/' + cls + '/' + fname)
                self.label_list.append(dataset.class_to_idx[cls])
    

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).long()


def get_transfer_datasets(dataset, data_root, data_split, n_shot=None, json_path=None):
    ds_name = dataset.lower()
    if ds_name in transfer_datasets.keys():
        # if ds_name == 'flower102':
        #     return torchvision.datasets.Flowers102(data_root, data_split)
        # elif ds_name == 'aircraft':
        #     return torchvision.datasets.FGVCAircraft(data_root, data_split)
        # elif ds_name == 'cifar10':
        #     return torchvision.datasets.CIFAR10(data_root, train=True if data_split=='train' else False)
        # elif ds_name == 'cifar100':
        #     return torchvision.datasets.CIFAR100(data_root, train=True if data_split=='train' else False)
        # else:
        #     # use custom dataloader
        d_set = eval(transfer_datasets[ds_name])
        return d_set(data_root, mode=data_split)
    elif ds_name == 'eurosat':
        # assert json_path is not None
        json_path_local="/cmlscratch/manlis/test/BackboneBenchmark/classification/datasets/data_splits/eurosat_split.json"
        return BaseJsonDataset(data_root, json_path_local, data_split, n_shot)
    elif ds_name == 'semiimagenet':
        return SemiImageNet(data_root, n_shot)
    else:
        raise NotImplementedError

if __name__=='__main__':
    dataset = get_transfer_datasets('flower102', '/cmlscratch/manlis/data/Flower-102', 'train')
    dataset.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    print("dataset loaded")
    print(f"dataset len: {len(dataset)}")

    # json_path_local="/cmlscratch/manlis/test/BackboneBenchmark/classification/datasets/data_splits/eurosat_split.json"
    # with open(json_path_local) as fp:
    #     splits = json.load(fp)
    #     samples = splits['train']
    #     import pdb
    #     pdb.set_trace()
    #     for s in samples:
    #         self.image_list.append(s[0])
    #         self.label_list.append(s[1])

    # loader = torch.utils.data.DataLoader(dataset)

    # for i, (x, y) in enumerate(loader):
    #     pass

    #train_files = urllib.request.urlopen(f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/10percent.txt').readlines()
    #samples = []
    #for fname in train_files:
    #    fname = fname.decode().strip()
    #    cls = fname.split('_')[0]
    #    import pdb
    #    pdb.set_trace()
    #    samples.append((IMAGENET_PATH + 'train/' + cls + '/' + fname, self.trainset.class_to_idx[cls]))
