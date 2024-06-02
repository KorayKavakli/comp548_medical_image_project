import torch
from models.model import Model
from models.train import train_model
from models.test import test_model
from utils.io_handler import save_model, load_model, write_accuracies
from utils.plot import plot_loss_curves, plot_confusion_matrix


def run(settings, device, train_loader, val_loader, test_loader, class_weights, mode="train"):
    model = Model(
        model_name=settings["model"]["name"],
        num_classes=settings["model"]["num classes"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["hyperparameters"]["learning rate"])
    #criterion = torch.nn.CrossEntropyLoss()
    if mode == "train":
        print("Training the model...")
        class_weights = class_weights.to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        model, train_loss, val_loss = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=settings["hyperparameters"]["epochs"],
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device
        )
        save_model(model, "{}/{}".format(
                                         settings["general"]["output directory"], 
                                         settings["model"]["name"]))
        plot_loss_curves(
                         train_loss, 
                         val_loss, 
                         settings["general"]["output directory"],
                         settings["model"]["name"])
    elif mode == "test":
        print("Testing the model...")
        model = load_model(model, "{}/{}".format(
                                                  settings["general"]["output directory"], 
                                                  settings["model"]["name"]))
        print("{} model loaded from {}/{}".format(
                                                  settings["model"]["name"], 
                                                  settings["general"]["output directory"],
                                                  settings["model"]["name"]))
        results = {
            "train": None,
            "validation": None,
            "test": None
        }
        loader_mapping = {
            train_loader: "train",
            val_loader: "validation",
            test_loader: "test"
        }
        criterion = torch.nn.CrossEntropyLoss()
        for loader in loader_mapping:
            if loader == test_loader:
                accuracy, predictions, labels = test_model(
                                                    model=model,
                                                    criterion=criterion,
                                                    test_loader=loader,
                                                    device=device,
                                                    num_classes=settings["model"]["num classes"]
                                                )
            else:
                accuracy, _, _ = test_model(
                                            model=model,
                                            criterion=criterion,
                                            test_loader=loader,
                                            device=device,
                                            num_classes=settings["model"]["num classes"]
                                        )
            results[loader_mapping[loader]] = accuracy
        write_accuracies(
                         results,    
                         "{}/{}".format(
                                        settings["general"]["output directory"], 
                                        settings["model"]["name"]))
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        plot_confusion_matrix(predictions,
                              labels,
                              settings["general"]["output directory"],
                              settings["model"]["name"],
                              settings["lesion types"])