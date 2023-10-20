from datasets.transfer_cls_datasets import *
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset

import wilds

from wilds import get_dataset
from datasets.inat_loader import INAT2019, INAT2021
import torchvision.transforms as transforms


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
    'eurosat': 'eurosat'
}


wilds_group_list = {
    'iwildcam': 'location',
    'fmow': 'region',
    'globalwheat': 'location',
    'camelyon17': 'hospital',
    'poverty': 'batch',
}

num_classes = {
    "inat2021": 10000,
    "inat2019": 1010,
    "imagenet": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "flower102": 102,
    "aircraft": 100,
    "eurosat": 10,
    "semiimagenet": 1000,
}


def create_other_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        seed=42,
        repeats=0,
        group=None,
        domain=None,
        json_path=None,
        n_shot=None,
        **kwargs
):
    """ Dataset for transfer learning and wilds
    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (HFDS, TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS)
        batch_size: batch size hint for (TFDS, WDS)
        seed: seed for iterable datasets (TFDS, WDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS, WDS)
        group: whether to specific a domain in wilds
        domain: which specific domain is used in wilds
        json_path: json file (of train/val split)
        n_shot: if integer: n_shot samples per class; if float: means the percentage
        **kwargs: other args to pass to dataset
    Returns:
        Dataset object
    """
    name = name.lower()
    
    if name in transfer_datasets:
        ds = get_transfer_datasets(name, root, 'train' if split=='train' else 'val', n_shot=n_shot, json_path=json_path)
    elif name in wilds.supported_datasets:
        dataset = get_dataset(dataset=name, download=False, root_dir=root, debug=True)
        if group is not None:
            group = wilds_group_list[name]
        ds = dataset.get_subset(split, group=group, domain=domain)
    elif name == 'inat2019':
        ds = INAT2019(root, mode='train' if split=='train' else 'val')
    elif name == 'inat2021':
        # root: 	/fs/vulcan-datasets/inat_comp_2021
        ds = INAT2021(root, 
            version='train' if split=='train' else 'val',
            target_type='full')
    else:
        print("Unknown dataset {}".format(name))

    return ds

if __name__=='__main__':
    print("debug")
    #dataset = create_other_dataset('pets', '/cmlscratch/manlis/data/Pets', split='test')
    dataset = create_other_dataset('iwildcam', '/fs/cml-datasets/WILDS/', split='train')
    #dataset = create_other_dataset('iwildcam', '/fs/cml-datasets/WILDS/', split='id_test')
    dataset.transform = transforms.Compose([
			transforms.RandomResizedCrop(224, interpolation=PIL.Image.BICUBIC),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
		])
    loader = torch.utils.data.DataLoader(dataset, batch_size = 100)

    for i, (x, y, metadata) in enumerate(loader):
        print(x.shape, y.shape)
        break
