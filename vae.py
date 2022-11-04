import torch.nn as nn
import torch

class VAE(nn.Module):
  def __init__(self, input_nodes, latent_nodes):
    super().__init__()
    self.inp = input_nodes
    self.lat = latent_nodes

    #encoder
    self.enc = nn.Sequential(
        nn.Linear(self.inp,self.lat),
        nn.ReLU(),
        nn.Linear(self.lat,self.lat*2)
    )

    #decoder
    self.dec = nn.Sequential(
        nn.Linear(self.lat,self.inp*2),
        nn.ReLU(),
        nn.Linear(self.inp*2,self.inp),
    )

  def reparametrise(self, mu, log_var):  
    std = torch.exp(0.5 * log_var) 
    eps = torch.randn_like(std)
    sample = mu + eps*std
    return sample

  def forward(self,x):
    # encoder
    x = self.enc(x).view(-1,2,self.lat)
    
    # reparametrise
    mu = x[:,0,:]
    log_var = x[:,1,:]
    z = self.reparametrise(mu, log_var)

    #decoder
    out = self.dec(z)

    return out, mu, log_var

  def generation(self, mu, log_var):
    #generate one data point
    z = self.reparametrise(mu, log_var)
    return self.dec(z)
