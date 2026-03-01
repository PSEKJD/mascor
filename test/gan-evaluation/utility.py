# %% Discriminative score
from .discriminator_model import *
from torch.utils.data import DataLoader
import copy
import numpy as np

def create_dl(real_dl, fake_dl,  hidden_dim, batch_size, dim):
    train_x, train_y = [], []
    for data in real_dl:
        train_x.append(data)
        train_y.append(torch.ones(data.shape[0],hidden_dim))
    for data in fake_dl:
        train_x.append(data)
        train_y.append(torch.zeros(data.shape[0],hidden_dim))
    x, y = torch.tensor(np.array(train_x)).view(-1,576,dim), torch.tensor(np.array(train_y)).view(-1,576,hidden_dim)
    print(x.shape, y.shape)
    idx = torch.randperm(x.shape[0])
    data_set = torch.cat((x[idx],y[idx]), dim = -1).to(torch.float32) # training only supports float32

    return DataLoader(data_set, batch_size=batch_size, shuffle = True)

# %%
def discriminative_score(train_dl, test_dl, epochs, device, hidden_dim, dim):
    dataloader = {'train': train_dl, 'validation': test_dl}
    input_size = dim
    model = Discriminator(input_size = input_size, hidden_dim = hidden_dim, num_layer = 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,)
    l_bce = nn.BCELoss()
    
    best_acc = 0.0
    best_loss = 999
    
    for epoch in range(epochs):
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0
            # iterate over data
            for data in dataloader[phase]:
                inputs = data[:,:,:input_size]
                labels = data[:,:,input_size:]
                labels = labels[:,0]
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = l_bce(outputs, labels)
                predictions = (outputs > 0.5).float()
                #_, preds = torch.max(outputs, 1)
                # BwrdPhase:
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item() * inputs.size(0)
                running_corrects +=  (predictions == labels).sum().item()
                total += labels.size(0)
                # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            
            print("Epoch loss_{epoch_loss} & Epoch_acc_{epoch_acc}".format(epoch_loss = epoch_loss, epoch_acc = epoch_acc))
            
            if phase == "validation" and epoch_acc >= best_acc:
                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()

    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    test_acc, test_loss = _test_classifier(
        model, test_dl, device,dim)
    return test_acc, best_acc

def _test_classifier(model, test_loader, device, dim):
    """
    Computes the test metric for trained classifier
    Parameters
    ----------
    model: torch.nn.module, trained model
    test_loader:  torch.utils.data DataLoader: dataset for testing
    config: configuration file

    Returns
    -------
    test_acc: model's accuracy in test dataset
    test_loss: model's cross-entropy loss in test dataset
    """
    # send model to device
    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0
    running_loss = 0
    criterion = nn.BCELoss()
    input_size = dim
    with torch.no_grad():
        # Iterate through data
        for data in test_loader:

            inputs = data[:,:,:input_size]
            labels = data[:,:,input_size:]
            labels = labels[:,0]
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predictions = (outputs > 0.5).float()

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    # Print results
    test_acc = correct / total
    test_loss = running_loss / total
    print("Accuracy of the network on the {} test samples: {}".format(total, (100 * test_acc)))
    return test_acc, test_loss

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(xx.shape),
                  torch.zeros(xx.shape))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
    return torch.mean(XX + YY - 2. * XY)