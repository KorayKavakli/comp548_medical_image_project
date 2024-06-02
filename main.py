import sys
from utils.helpers import seed_everything, set_device
from utils.io_handler import read_settings, check_dir
from dataloader.dataloader import prepare_dataloaders
from models.run import run


__author__ = "Koray Kavakli"
__title__ = "Class project for medical image analysis"


def prepare():
    settings, mode, filename = read_settings(title=__title__)
    print("Running in {} with settings read from {}.".format(mode, filename))    
    device = set_device(settings["general"]["device"])
    output_dir = settings["general"]["output directory"]
    lesion_type_dict = settings["lesion types"]
    seed_everything(settings["general"]["seed"])
    check_dir(output_dir)
    return settings, device, lesion_type_dict, mode


def main(): 
    settings, device, lesion_type_dict, mode = prepare() 

    train_loader, val_loader, test_loader, class_weights = prepare_dataloaders(
        settings=settings,
        split_ratios=settings["dataset"]["split ratios"],
        lesion_type_dict=lesion_type_dict
    )

    if mode == "train":
        run(settings, device, train_loader, val_loader, test_loader, class_weights, mode="train")
    elif mode == "test":
        run(settings, device, train_loader, val_loader, test_loader, class_weights, mode="test")
    else:
        raise ValueError("Mode is not valid. Please choose from 'train', 'test'.")


if __name__ == '__main__':
    sys.exit(main())