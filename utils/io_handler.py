import argparse
import os
import json
import torch


def read_settings(title="."):
    """
    Function to read the settings from a json file.

    Parameters:
    -----------
    filename          : str
                        Filename of the json file.
    title             : str
                        Title of the description of the parser.

    Returns:
    --------
    settings          : dict
                        Dictionary with the settings.
    """
    default_filename = "./settings/sample_settings.txt"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument(
                        "--settings", 
                        type=argparse.FileType('r'), 
                        default=default_filename, 
                        help="Filename for the settings file. Default is ()".format(default_filename)
                       )
    parser.add_argument(
                        "--mode", 
                        type=str, 
                        default="train", 
                        help="Mode to run the model. Default is 'train'."
                       )
    args = parser.parse_args()
    if type(args.settings) != type(None):
        filename = args.settings.name
    else:
        filename = default_filename
    if type(args.mode) != type(None):
        mode = str(args.mode)
    settings = json.load(open((filename)))
    return settings, mode, filename


def check_dir(directory):
    """
    Function to check if a directory exists and if not, create it.

    Parameters:
    -----------
    directory         : str
                        Directory to check.

    Returns:
    --------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, directory):
    """
    Function to save the model.

    Parameters:
    -----------
    model             : torch model
                        Model to save.
    directory         : str
                        Directory to save the model.

    Returns:
    --------
    None
    """
    check_dir(directory)
    torch.save(
               model.state_dict(), 
               "{}/{}".format(directory, "model.pth")
              )
    

def load_model(model, directory):
    """
    Function to load the model.

    Parameters:
    -----------
    model             : torch model
                        Model to load.
    directory         : str
                        Directory to load the model.

    Returns:
    --------
    model             : torch model
                        Model to be used.
    """
    model.load_state_dict(torch.load("{}/{}".format(directory, "model.pth")))
    return model
    

def write_accuracies(accuracy_dicts, folder, filename="accuracies.txt"):
    """
    Function to write accuracies to a file.

    Parameters
    ----------
    accuracies          : dict
                          key value mapping of accuracies.
    folder              : str
                          folder to store the file.
    filename            : str
                          filename to store the accuracies.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'w') as f:
        for dataset, accuracies in accuracy_dicts.items():
            f.write(f"{dataset} dataset accuracies:\n")
            for label, accuracy in accuracies.items():
                f.write(f"{label}: {accuracy}\n")
            f.write("\n")
