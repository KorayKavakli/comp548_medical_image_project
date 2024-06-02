import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(Model, self).__init__()
        """
        Class to define the model.
        
        Parameters:
        -----------
        model_name          : str
                              Name of the model.
        num_classes         : int
                              Number of classes.
        """
        self.num_classes = num_classes
        if model_name == "AlexNet":
            self.model = self.alexnet(num_classes, pretrained=True)
        elif model_name == "ResNet18":
            self.model = self.resnet18(num_classes, pretrained=True)
        elif model_name == "VGG11":
            self.model = self.vgg11(num_classes, pretrained=True)
        else:
            raise ValueError("Model name is not valid. Please choose from 'AlexNet', 'ResNet18', 'VGG11'.")


    def alexnet(self, num_classes, pretrained=True):
        """
        Function to load the AlexNet model.
        
        Parameters:
        -----------
        num_classes         : int
                              Number of classes.
        pretrained          : bool
                              Whether to load the pretrained model or not.
                                
        Returns:
        --------
        model               : torch model
                              Model to be used.
        """
        model = torchvision.models.alexnet(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        return model


    def resnet18(self, num_classes, pretrained=True):
        """
        Function to load the ResNet18 model.

        Parameters:
        -----------
        num_classes         : int
                              Number of classes.
        pretrained          : bool
                              Whether to load the pretrained model or not.

        Returns:
        --------
        model               : torch model
                              Model to be used.
        """                          
        model = torchvision.models.resnet18(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model


    def vgg11(self, num_classes, pretrained=True):
        """
        Function to load the VGG19 model.

        Parameters:
        -----------
        num_classes         : int
                              Number of classes.
        pretrained          : bool
                              Whether to load the pretrained model or not.

        Returns:
        --------
        model               : torch model
                              Model to be used.
        """
        model = torchvision.models.vgg11(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        x             : tensor
                        Input tensor.
        
        Returns:
        --------
        x             : tensor
                        Output tensor.
        """
        return self.model(x)