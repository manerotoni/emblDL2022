import os
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from imageio import imread
from tqdm import tqdm, trange

def normalize(x, pmin=1, pmax=99.8):
    mi, ma = np.percentile(x, (pmin, pmax))
    eps = 1.e-6
    x = x.astype(np.float32)
    x = (x-mi)/(ma-mi+eps) 
    return x

def pad_to_shape(d, dshape, mode = "constant"):
    """
    pad array d to shape dshape
    """
    if d.shape == dshape:
        return d

    diff = np.array(dshape) - np.array(d.shape)
    #first shrink
    slices  = tuple(slice(-x//2,x//2) if x<0 else slice(None,None) for x in diff)
    res = d[slices]
    #then pad
    return np.pad(res,tuple((int(np.ceil(d/2.)),d-int(np.ceil(d/2.))) if d>0 else (0,0) for d in diff),mode=mode)

def save_checkpoint(model, optimizer, epoch, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)


def load_checkpoint(save_path, model, optimizer):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


#
# training and validation functions
#


def get_current_lr(optimizer):
    lrs = [param_group.get('lr', None) for param_group in optimizer.param_groups]
    lrs = [lr for lr in lrs if lr is not None]
    # to keep things simple we only return one of the valid lrs
    return lrs[0]


def train(model, loader,
          loss_function, optimizer,
          device, epoch,
          tb_logger, log_image_interval=100):
    """ Train model for one epoch.

    Parameters:
    model - the model we are training
    loader - the data loader that provides the training data
        (= pairs of images and labels)
    loss_function - the loss function that will be optimized
    optimizer - the optimizer that is used to update the network parameters
        by backpropagation of the loss
    device - the device used for training. this can either be the cpu or gpu
    epoch - which trainin eppch are we in? we keep track of this for logging
    tb_logger - the tensorboard logger, it is used to communicate with tensorboard
    log_image_interval - how often do we send images to tensborboard?
    """

    # set model to train mode
    model.train()

    n_batches = len(loader)

    # log the learning rate before the epoch
    lr = get_current_lr(optimizer)
    tb_logger.add_scalar(tag='learning-rate',
                         scalar_value=lr,
                         global_step=epoch * n_batches)

    # iterate over the training batches provided by the loader
    for batch_id, (x, y) in enumerate(loader):

        # send data and target tensors to the active device
        x = x.to(device)
        y = y.to(device)

        # set the gradients to zero, to start with "clean" gradients
        # in this training iteration
        optimizer.zero_grad()

        # apply the model to get the prediction
        prediction = model(x)

        # calculate the loss (negative log likelihood loss)
        # the loss function expects a 1d tensor, so we get rid of the second
        # singleton dimensions that is added by the loader when stacking across the batch function
        loss_value = loss_function(prediction, y)

        # calculate the gradients (`loss.backward()`)
        # and apply them to the model parameters according
        # to our optimizer (`optimizer.step()`)
        loss_value.backward()
        optimizer.step()

        # log the loss value to tensorboard
        step = epoch * n_batches + batch_id
        tb_logger.add_scalar(tag='train-loss',
                             scalar_value=loss_value.item(),
                             global_step=step)


def validate(model, loader, loss_function,
             device, step, tb_logger=None):
    """
    Validate the model predictions.

    Parameters:
    model - the model to be evaluated
    loader - the loader providing images and labels
    loss_function - the loss function
    device - the device used for prediction (cpu or gpu)
    step - the current training step. we need to know this for logging
    tb_logger - the tensorboard logger. if 'None', logging is disabled
    """
    # set the model to eval mode
    model.eval()
    n_batches = len(loader)

    # we record the loss and the predictions / labels for all samples
    mean_loss = 0
    predictions = []
    labels = []

    # the model parameters are not updated during validation,
    # hence we can disable gradients in order to save memory
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)

            # update the loss
            # the loss function expects a 1d tensor, so we get rid of the second
            # singleton dimensions that is added by the loader when stacking across the batch function
            mean_loss += loss_function(prediction, y).item()

            # compute the most likely class predictions
            # note that 'max' returns a tuple with the
            # index of the maximun value (which correponds to the predicted class)
            # as second entry
            prediction = prediction.max(1)[1]

            # store the predictions and labels
            predictions.append(prediction.to('cpu').numpy())
            labels.append(y.to('cpu').numpy())

    # predictions and labels to numpy arrays
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # log the validation results if we have a tensorboard
    if tb_logger is not None:

        accuracy_error = 1. - metrics.accuracy_score(labels, predictions)
        mean_loss /= n_batches

        # TODO log more advanced things like confusion matrix, see
        # https://www.tensorflow.org/tensorboard/image_summaries

        tb_logger.add_scalar(tag="validation-error",
                             global_step=step,
                             scalar_value=accuracy_error)
        tb_logger.add_scalar(tag="validation-loss",
                             global_step=step,
                             scalar_value=mean_loss)

    # return all predictions and labels for further evaluation
    return predictions, labels


def run_training(model, optimizer,
    train_loader, val_loader,
    device, name, n_epochs, class_weights=None):
    """ Complete training logic
    """

    if class_weights is not None:
        class_weights = torch.tensor(class_weights).float()
    
    best_accuracy = 0.

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    loss_function.to(device)

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='max',
                                  factor=0.5,
                                  patience=50)

    checkpoint_path = f'best_checkpoint_{name}.tar'
    log_dir = f'runs/{name}'
    tb_logger = SummaryWriter(log_dir)

    t = trange(n_epochs)
    for epoch in t:
        train(model, train_loader, loss_function, optimizer,
              device, epoch, tb_logger=tb_logger)
        step = (epoch + 1) * len(train_loader)

        pred, labels = validate(model, val_loader, loss_function,
                                device, step,
                                tb_logger=tb_logger)
        val_accuracy = metrics.accuracy_score(labels, pred)
        scheduler.step(val_accuracy)

        # otherwise, check if this is our best epoch
        if val_accuracy > best_accuracy:
            # if it is, save this check point
            best_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            
        t.set_description(f"Validation acc {val_accuracy}")

    tb_logger.close()
    
    return checkpoint_path

