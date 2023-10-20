from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms

from wilds.common.grouper import CombinatorialGrouper

dataset = get_dataset(dataset="iwildcam", download=False, root_dir='/fs/cml-datasets/WILDS/')

grouper = CombinatorialGrouper(dataset, ['location'])

# Get the training set
#train_data = dataset.get_subset(
#                "train",
#                transform=transforms.Compose(
#                [transforms.Resize((448, 448)), transforms.ToTensor()]
#                ),
#)

# Prepare the standard data loader
#train_loader = get_train_loader("standard", train_data, batch_size=16)
#train_loader = get_train_loader("group", train_data, grouper=grouper, n_groups_per_batch=1, batch_size=16)

#for x_test, y_true, metadata in train_loader:
#    z_test = grouper.metadata_to_group(metadata)

# (Optional) Load unlabeled data
#dataset = get_dataset(dataset="iwildcam", download=True, unlabeled=True)
#unlabeled_data = dataset.get_subset(
#            "test_unlabeled",
#            transform=transforms.Compose(
#            [transforms.Resize((448, 448)), transforms.ToTensor()]
#            ),
#)
#unlabeled_loader = get_train_loader("standard", unlabeled_data, batch_size=16)

# Train loop
#for labeled_batch, unlabeled_batch in zip(train_loader, unlabeled_loader):
#    x, y, metadata = labeled_batch
#    unlabeled_x, unlabeled_metadata = unlabeled_batch
