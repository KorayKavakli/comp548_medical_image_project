import torch


def test_model(model, criterion, test_loader, device, num_classes):
    """
    Function to test the model.

    Parameters:
    -----------
    model               : torch model
                          Model to be used.
    criterion           : torch loss
                          Loss function to be used.
    test_loader         : torch dataloader
                          Dataloader for testing.
    device              : torch device
                          Device to be used.
    num_classes         : int
                          Number of classes.

    Returns:
    --------
    test_loss           : float
                          Loss for the test set.
    test_accuracy       : float
                          Accuracy for the test set.
    """
    model.eval()
    num_corrects_per_class = torch.zeros(num_classes)
    num_samples_per_class = torch.zeros(num_classes)
    predictions_list = []
    labels_list = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            predictions_list.append(preds)
            labels_list.append(labels)
        for i in range(len(labels)):
            num_samples_per_class[labels[i]] += 1
            if preds[i] == labels[i]:
                num_corrects_per_class[labels[i]] += 1
    accuracy_dict = {}
    for i in range(num_classes):
        accuracy_dict[f"Class {i + 1}"] = num_corrects_per_class[i].item() / num_samples_per_class[i].item()
    accuracy_dict["Overall"] = torch.sum(num_corrects_per_class).item() / torch.sum(num_samples_per_class).item()
    return accuracy_dict, predictions_list, labels_list