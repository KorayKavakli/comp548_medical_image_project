import numpy as np
import torch
import matplotlib.pyplot as plt
import scienceplots
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix



def plot_loss_curves(training_loss, validation_loss, directory, model):
    """
    Function to plot loss curves.

    Parameters:
    -----------
    training_loss         : list
                            Training loss.
    validation_loss       : list
                            Validation loss.    
    directory             : str
                            Directory to save the plots.        
    title                 : str
                            Title for the plot. 
    model                 : str
                            Model name.

    Returns:
    --------
    None
    """
    with plt.style.context(['science', 'ieee', 'grid']):
        plt.figure(figsize=(4, 4))
        plt.plot(training_loss, label="Training Loss")
        plt.plot(validation_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, len(training_loss))
        plt.ylim(0, max(training_loss) + 0.2)
        plt.title("Loss Curves - {}".format(model))
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}/{}/{}".format(directory, model, "loss_curves.png"),
                     dpi=300, 
                     bbox_inches="tight")
        plt.close()


def plot_confusion_matrix(predictions, labels, directory, model, lesion_types, normalize=True):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    matrix = confusion_matrix(labels, predictions)
    class_labels = lesion_types
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = plt.cm.Blues if not normalize else plt.cm.viridis
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_labels)
        disp.plot(ax=ax, cmap=cmap)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.minorticks_off()
        ax.tick_params(axis='x',which='major', direction="out", bottom=True, top=False)
        ax.tick_params(axis='y',which='major', direction="out", left=True, right=False)
        plt.grid(False)
        plt.title("Confusion Matrix - {}".format(model))
        
        plt.tight_layout()
        plt.savefig("{}/{}/{}".format(directory, model, "confusion_matrix_test_dataset.png"),
                    dpi=300, 
                    bbox_inches="tight")
        plt.close()