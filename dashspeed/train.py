import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import SpeedVideoSampler
from .model import initialize_model
from .params import *


def train_model(model, dataset, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in dataset:
                frame1, frame2, speed = batch
                frame1 = frame1.to(device)
                frame2 = frame2.to(device)
                speed = speed.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception:
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(frame1, frame2)
                        loss1 = criterion(outputs, speed)
                        loss2 = criterion(aux_outputs, speed)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(frame1, frame2)
                        loss = criterion(outputs, speed)

                    _, preds = torch.max(outputs, 1)
                    preds = outputs
                    loss.backward()
                    optimizer.step()

                _loss = loss.item()

                # statistics
                running_loss += _loss * frame1.size(0)

                print('loss: {:.4f} delta speed: {:.4f}'.format(_loss, _loss ** .5 * dataset.dataset.max_speed))

            epoch_loss = running_loss / len(dataset.dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        print()
    finally:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), './best_model.pth')
    return model, val_acc_history


def run_training_routine():
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)

    print("Initializing Datasets and Dataloaders...")

    train_dataset = SpeedVideoSampler('./data')
    sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(train_dataset), batch_size=batch_size, drop_last=True)
    batched_dataset = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler)

    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    ######################################################################
    # Run Training and Validation Step
    # --------------------------------
    #
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.
    #

    # Setup the loss fxn
    criterion = nn.MSELoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, batched_dataset, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    return model_ft, hist
