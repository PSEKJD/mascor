# discriminative models
import torch
import torch.nn as nn
import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
      for name,param in m.named_parameters():
        if 'weight_ih' in name:
          init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
          init.orthogonal_(param.data)
        elif 'bias' in name:
          param.data.fill_(0)
          
class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    def __init__(self, input_size, hidden_dim, num_layer):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layer, batch_first=True)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        _, d_outputs = self.rnn(input)
        Y_hat = self.fc(d_outputs[0])
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat

class Discriminator_v2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator_v2, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, t):
        _, h_n = self.gru(x)
        y_hat_logit = self.fc(h_n).squeeze()
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat