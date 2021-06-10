from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_image_dataset(project, preprocessing, signal_length, figtype, image_size=224):

    fig_path = (project.images_dir / preprocessing /
                figtype / signal_length).__str__()

    dataset = datasets.ImageFolder(root=fig_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))
                                   ])
                                   )
    return dataset


def get_image_dataloaders(project, preprocessing, signal_length, figtype, image_size=224,
                          train_split=0.8, batch_size=8, workers=1, ngpu=1, random_seed=42):

    dataset = get_image_dataset(project, preprocessing, signal_length, figtype)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    trainsplit = int(np.floor(train_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[:trainsplit], indices[trainsplit:]
    trainset_size = len(train_indices)
    valset_size = len(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=workers)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,
                                   num_workers=workers)

    dataloaders = {"train": train_loader, "val": validation_loader}

    dataset_sizes = {"train": len(train_indices), "val": len(val_indices)}

    return dataloaders, dataset_sizes
