"""
Transfer learn driving speed
============================

**Adapted From:** https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
import time
import os
import copy

from .params import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


######################################################################
# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


######################################################################
# Initialize and Reshape the Networks
# -----------------------------------

class SpeedModel(nn.Module):
    '''
    Supply a convolutional image model to identify features
    Make a trainable encoder to learn position identifiers
    Merge two encoded images to an FC to output speed
    '''
    def __init__(self, image_model):
        super(SpeedModel, self).__init__()
        self.image_model = image_model
        self.conv = nn.Conv2d(256, 12, 1)
        self.position_enc = nn.Linear(3192, 32)#12*29*39, 32)
        self.fc1 = nn.Linear(32*2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, frame1, frame2):
        positional_features = []
        flatten = lambda x : x.view(x.size()[0], -1)
        for frame in [frame1, frame2]:
            x = self.image_model(frame)
            x = F.relu(self.conv(x))
            x = self.position_enc(flatten(x))
            positional_features.append(x)
        x = torch.cat(positional_features, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft = model_ft.features
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft = model_ft.features
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_tf = model_ft.features
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()


    return SpeedModel(model_ft), input_size
