import torch
import torchvision.transforms
from dataloader.dataset import HAM10000


def set_transforms(normalize, resize, mean, std):
    """
    Function to set the transforms for the dataset.
    
    Parameters:
    -----------
    normalize   : bool
                  Whether to normalize the images or not.
    resize      : int
                  Size to resize the images to.
    mean        : list
                  Mean for normalization.
    std         : list
                  Standard deviation for normalization.
                  
    Returns:
    --------
    transforms  : torchvision.transforms
                  Transforms to be used.
    """
    base_transforms = [torchvision.transforms.Resize(resize), torchvision.transforms.ToTensor()]
    if normalize:
        base_transforms.append(torchvision.transforms.Normalize(mean=mean, std=std))
    transforms = torchvision.transforms.Compose(base_transforms)
    return transforms


def prepare_dataloaders(settings, split_ratios, lesion_type_dict):
    """
    Function to prepare the dataloaders.

    Parameters:
    -----------
    settings          : dict
                        Settings dictionary.
    split_ratios      : list
                        List of split ratios.
    lesion_type_dict  : dict
                        Dictionary of lesion types. 

    Returns:
    --------
    train_loader      : torch dataloader
                        Dataloader for training.
    val_loader        : torch dataloader
                        Dataloader for validation.
    test_loader       : torch dataloader    
                        Dataloader for testing.
    """
    transforms = set_transforms(
        normalize=settings["model"]["normalize"],
        resize=settings["model"]["resize"],
        mean=settings["model"]["mean"],
        std=settings["model"]["std"]
    )
    full_dataset = HAM10000(
        dataset_dir=settings["dataset"]["dataset directory"],
        transform=transforms,
        lesion_type_dict=lesion_type_dict
    )
    total_size = len(full_dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=settings["hyperparameters"]["batch size"],
        shuffle=False,
        num_workers=settings["hyperparameters"]["num workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=settings["hyperparameters"]["batch size"],
        shuffle=False,
        num_workers=settings["hyperparameters"]["num workers"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=settings["hyperparameters"]["batch size"],
        shuffle=False,
        num_workers=settings["hyperparameters"]["num workers"]
    )
    return train_loader, val_loader, test_loader