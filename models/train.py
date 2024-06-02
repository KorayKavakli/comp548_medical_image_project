import torch
import tqdm
import copy


def train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, class_weights, device):
    """
    Function to train the model.
    
    Parameters:
    -----------
    model               : torch model
                          Model to be trained.
    criterion           : torch loss
                          Loss function.
    optimizer           : torch optimizer
                          Optimizer to be used.
    num_epochs          : int
                          Number of epochs.   
    train_loader        : torch dataloader
                          Dataloader for training.    
    val_loader          : torch dataloader
                          Dataloader for validation.
    device              : torch device
                          Device to be used.

    Returns:
    --------
    model                     : torch model
                                Trained model.
    training_loss_per_epoch   : list
                                Training loss per epoch.
    validation_loss_per_epoch : list
                                Validation loss per epoch.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_no_corrects = 0
    training_loss_per_epoch = []
    validation_loss_per_epoch = []
    training_bar = tqdm.tqdm(range(num_epochs), desc="Training")
    for epoch in training_bar:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                #print(loss)
                loss = loss * class_weights[labels]
                #print(class_weights[labels])
                #print(class_weights[labels].shape)
                #print(loss)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        training_loss_per_epoch.append(epoch_loss / num_batches)
        model.eval()
        no_corrects = 0
        num_batches = 0
        epoch_loss = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                no_corrects += torch.sum(preds == labels.data)
                loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            num_batches += 1
        validation_loss_per_epoch.append(epoch_loss / num_batches)
        if no_corrects > best_no_corrects:
            best_no_corrects = no_corrects
            best_model_wts = copy.deepcopy(model.state_dict())
        training_bar.set_description(f"Training Loss: {training_loss_per_epoch[-1]:.4f} Validation Loss: {validation_loss_per_epoch[-1]:.4f}")
    model.load_state_dict(best_model_wts)
    return model, training_loss_per_epoch, validation_loss_per_epoch