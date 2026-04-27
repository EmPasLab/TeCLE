import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from torch_ac.torch_ac.algos import common

from torch_ac.torch_ac.algos import colorednoise


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        
        c, h, w = shape
        h, w = common.calc_conv2d_output((h, w), 1, 1, 0)
        conv2d_out_size = 64 * h * w
        
        self.encode = nn.Sequential(nn.Conv2d(c, 32, 1, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True),
                                    nn.Conv2d(32, 64, 1, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, 1, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True),
                                    Flatten(),
                                    MLP([conv2d_out_size, 256, 128])
                                   )
        self.calc_mean = MLP([128+ncond, 64, nhid], last_activation = False)
        self.calc_logvar = MLP([128+ncond, 64, nhid], last_activation = False)


    def forward(self, x, y = None):
        
        x = self.encode(x)
        
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))

        
class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid+ncond, 64, 128, 256, c*w*h], last_activation = False), nn.Sigmoid())

    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)


class InverseNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(InverseNet, self).__init__()
        h, w, c = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)

        conv2d_out_size = 32 * h * w
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flat = nn.Flatten()

        self.inverse_net = nn.Sequential(
            nn.Linear(conv2d_out_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        common.initialize_weights(self)

    def forward(self, s_t, s_tm1):
        features_tm1 = self.body(s_tm1)
        flat_features_tm1 = self.flat(features_tm1)
        features_t = self.body(s_t)
        flat_features_t = self.flat(features_t)

        inverse_input = torch.cat([flat_features_tm1, flat_features_t], dim=-1)
        pi_logits = self.inverse_net(inverse_input)

        return pi_logits, features_t
    
    
class cVAE(nn.Module):
    def __init__(self, device, state_dim, nclass, nhid = 16, ncond = 16, noise_beta=0.0):
        super(cVAE, self).__init__()
        self.dim = nhid
        self.device = device
        h, w, c = state_dim
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        h, w = common.calc_conv2d_output((h, w), 3, 1, 1)
        body_out_size = (32, h, w)

        
        self.encoder = Encoder(body_out_size, nhid, ncond = ncond)
        self.decoder = Decoder(body_out_size, nhid, ncond = ncond) 

        common.initialize_weights(self)

        self.label_embedding = nn.Embedding(nclass, ncond)         
        self.noise_beta = noise_beta
        
    def generate_noise(self, size):
        noise = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size)
        return torch.tensor(noise, dtype=torch.float32).to(self.device)
    
    def sampling(self, mean, logvar):
        if self.noise_beta == 0:
            eps = torch.randn(mean.shape).to(self.device)
        else:
            eps = self.generate_noise(mean.shape)
        sigma = torch.exp(0.5 * logvar)
        return mean + eps * sigma
        

    def forward(self, features_t, a_tm1):
        
        y = self.label_embedding(a_tm1)
        mean, logvar = self.encoder(features_t, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar


cnt = 0
def cvae_loss(X, X_hat, mean, logvar):

    X = F.sigmoid(X)
    reconstruction_loss = F.binary_cross_entropy(X_hat, X)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mean**2)
    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self, device, shape, nhid = 16):
        super(VAE, self).__init__()
        self.dim = nhid
        self.device = device
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar
    
    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)).to(self.device) if batch_size else torch.randn((1, self.dim)).to(self.device)
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res
