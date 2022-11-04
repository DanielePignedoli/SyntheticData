import torch

class Discriminator(torch.nn.Module):
  def __init__(self, hidden_nodes):
    super(Discriminator, self).__init__()
    self.seq = torch.nn.Sequential(
        torch.nn.Linear(4+1,hidden_nodes),
        torch.nn.LeakyReLU(negative_slope=0.2),
        torch.nn.Linear(hidden_nodes,1),
        torch.nn.Sigmoid()
    )
    self.emb =  torch.nn.Embedding(3,1) #vars: num of embeddings (so num of species), dimension of embedding vector

  def forward(self,x,labels):
    embedding = self.emb(labels)
    x = torch.cat([x,embedding], dim = 1)
    return self.seq(x)

class Generator(torch.nn.Module):
  def __init__(self, noise_dim, hidden_nodes):
    super(Generator, self).__init__()

    self.noise_dim = noise_dim

    self.seq = torch.nn.Sequential(
        torch.nn.Linear(noise_dim+1,hidden_nodes),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_nodes,4),
        torch.nn.Tanh()
    )
    self.emb =  torch.nn.Embedding(3,1)

  def forward(self,x,labels):
    embedding = self.emb(labels)
    x = torch.cat([x,embedding], dim = 1)
    return self.seq(x)


