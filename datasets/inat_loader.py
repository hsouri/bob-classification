import torch
import torch.utils.data as data
from PIL import Image
import os
import json
import torchvision
from torchvision import transforms
import random
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import PIL
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INAT2019(data.Dataset):
    def __init__(self, root, mode='train', year="2019", transform=None):
        # load annotations
        ann_file = os.path.join(root, f"{mode}{year}.json")
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = []
            for i, aa in enumerate(ann_data['annotations']):
                assert aa['image_id'] == self.ids[i]
                self.classes.append(aa['category_id'])
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        # self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        # self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.loader = default_loader

        self.transform = transform
        # # augmentation params
        # self.im_size = [299, 299]  # can change this to train on higher res
        # self.mu_data = [0.485, 0.456, 0.406]
        # self.std_data = [0.229, 0.224, 0.225]
        # self.brightness = 0.4
        # self.contrast = 0.4
        # self.saturation = 0.4
        # self.hue = 0.25

        # # augmentations
        # self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        # self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        # self.flip_aug = transforms.RandomHorizontalFlip()
        # self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        # self.tensor_aug = transforms.ToTensor()
        # self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.imgs[index])
        img = self.loader(path)
        species_id = self.classes[index]
        # tax_ids = self.classes_taxonomic[species_id]

        # if self.is_train:
        #     img = self.scale_aug(img)
        #     img = self.flip_aug(img)
        #     img = self.color_aug(img)
        # else:
        #     img = self.center_crop(img)

        # img = self.tensor_aug(img)
        # img = self.norm_aug(img)
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(species_id).long()

    def __len__(self):
        return len(self.imgs)
    

CATEGORIES_2021 = ["kingdom", "phylum", "class", "order", "family", "genus"]

class INAT2021(torchvision.datasets.VisionDataset):
    """
    modified from torchvision.datasets.INaturalist() to work with vulcan inat2021
    """

    def __init__(
        self,
        root: str,
        version: str = "train",
        target_type: Union[List[str], str] = "full",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.version = version
        super().__init__(os.path.join(root, version), transform=transform, target_transform=target_transform)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        if not isinstance(target_type, list):
            target_type = [target_type]

        self.target_type = target_type
        self._init_2021()


        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.all_categories):
            files = os.listdir(os.path.join(self.root, dir_name))
            for fname in files:
                self.index.append((dir_index, fname))

    def _init_2021(self) -> None:
        """Initialize based on 2021 layout"""

        self.all_categories = sorted(os.listdir(self.root))

        # map: category type -> name of category -> index
        self.categories_index = {k: {} for k in CATEGORIES_2021}

        for dir_index, dir_name in enumerate(self.all_categories):
            pieces = dir_name.split("_")
            if len(pieces) != 8:
                raise RuntimeError(f"Unexpected category name {dir_name}, wrong number of pieces")
            if pieces[0] != f"{dir_index:05d}":
                raise RuntimeError(f"Unexpected category id {pieces[0]}, expecting {dir_index:05d}")
            cat_map = {}
            for cat, name in zip(CATEGORIES_2021, pieces[1:7]):
                if name in self.categories_index[cat]:
                    cat_id = self.categories_index[cat][name]
                else:
                    cat_id = len(self.categories_index[cat])
                    self.categories_index[cat][name] = cat_id
                cat_map[cat] = cat_id
            self.categories_map.append(cat_map)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(os.path.join(self.root, self.all_categories[cat_id], fname))

        target: Any = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise ValueError(f"Invalid category type '{category_type}'")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise ValueError(f"Invalid category id {category_id} for {category_type}")

    def _check_integrity(self) -> bool:
        return os.path.exists(self.root) and len(os.listdir(self.root)) > 0


if __name__=='__main__':
    print("debug")
    #dataset = create_other_dataset('pets', '/cmlscratch/manlis/data/Pets', split='test')
    dataset = INAT2019('/fs/cml-projects/benchmarking_backbone/dataset/inat2019', mode='val')
    print("num classes: ", max(dataset.classes))

    dataset.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    print("dataset loaded")

    loader = torch.utils.data.DataLoader(dataset)

    for i, (x, y) in enumerate(loader):
        pass