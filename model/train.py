
""" train nICA model """

import copy
import numpy as np

import torch
import torch.nn as nn


# =============================================================================
# =============================================================================
def _do_train(model, loader, optimizer, criterion, device):
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    for idx_batch, (batch_x, batch_y, batch_z) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        batch_z = batch_z.to(device=device, dtype=torch.int32)
        y = [batch_y[(batch_z == k).nonzero().squeeze()] for k in torch.unique(batch_z)]
        batch_y = torch.concatenate(y,axis=0)
        
        logits, _ = model(batch_x, batch_z)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        train_loss[idx_batch] = loss.item()
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'
        
    return np.mean(train_loss)

# =============================================================================
# =============================================================================
def _validate(model, loader, criterion, device):
    # validation loop
    val_loss = np.zeros(len(loader))
    accuracy = 0.
    with torch.no_grad():
        model.eval()

        for idx_batch, (batch_x, batch_y, batch_z) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            batch_z = batch_z.to(device=device, dtype=torch.int32)
            y = [batch_y[(batch_z == k).nonzero().squeeze()] for k in torch.unique(batch_z)]
            batch_y = torch.concatenate(y,axis=0)
            
            output, _ = model.forward(batch_x, batch_z)

            loss = criterion(output, batch_y)
            val_loss[idx_batch] = loss.item()

            _, top_class = output.topk(1, dim=1)
            top_class = top_class.flatten()
            accuracy += \
                torch.sum((batch_y == top_class).to(torch.float32))


    accuracy = accuracy / len(loader.dataset)
    #print("---  Accuracy : %s" % accuracy.item(), "\n")
    return np.mean(val_loss), accuracy.item()

# =============================================================================
# =============================================================================
def train(model, loader_train, loader_valid, optimizer, scheduler, n_epochs,
          device):
    """Training function

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    device : str |Â instance of torch.device
        The device to train the model on.

    Returns
    -------
    best_model : instance of nn.Module
        The model that lead to the best prediction on the validation
        dataset.
    """
    # put model on cuda if not already
    device = torch.device(device)
    model.to(device)

    # define criterion
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = copy.deepcopy(model)
    
    train_loss = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
        train_loss[epoch] = _do_train(model, loader_train, optimizer, criterion, device)
        val_loss[epoch], val_accu = _validate(model, loader_valid, criterion, device)
        scheduler.step()
        format_str = ('Epoch: %d/%d.. Training Loss: %.2f.. Test Loss: %.2f.. Test Accuracy: %3.2f')
        print(format_str % (epoch, n_epochs, train_loss[epoch], val_loss[epoch], val_accu*100))

        # model saving
        if val_accu > best_val_acc:
            #print("\nbest val loss {:.4f} -> {:.4f}".format(best_val_loss, np.mean(val_loss)))
            best_val_acc = val_accu
            best_model = copy.deepcopy(model)
            
    return best_model, train_loss, val_loss

