import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class my_dataset():
    
  def __init__(self, data, transform = False):
    self.iris = data
    if transform:
        self.sc = MinMaxScaler((-1,1))
        self.iris[self.iris.columns[:-1]] = self.sc.fit_transform(self.iris[self.iris.columns[:-1]])
    xy = self.iris.values
    self.x = torch.tensor(xy[:,:4],  dtype =torch.float32)
    self.y = torch.tensor(xy[:,-1], dtype = torch.int64)
    
  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self,index):
    return self.x[index], self.y[index]
    
  def getmap(self):
      return self.mapp
  
  def inverse_scaling(self, array):
      return self.sc.inverse_transform(array)