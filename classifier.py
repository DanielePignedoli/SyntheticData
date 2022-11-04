import torch
from my_iris_dataset import my_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns


class Classify(torch.nn.Module):
  def __init__(self):
    super(Classify,self).__init__()

    self.seq = torch.nn.Sequential(
        torch.nn.Linear(4,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,3),
        torch.nn.Softmax(dim=1)  #softmax gives a list of probabilities
    )

  def forward(self,x):
    return self.seq(x)

#prepare data
iris = sns.load_dataset('iris')
mapp = {'setosa' : 0, 'versicolor' : 1, 'virginica': 2}
iris.species = iris.species.apply(lambda x : mapp[x])
data = my_dataset(data = iris)
data.y = torch.nn.functional.one_hot(data.y) 
data.y = data.y.type(torch.float32)
    
# hyperparams
batch_size = 10
lr = 0.005
max_epoch = 50


# loader
train, test = train_test_split(data)
train_ld = DataLoader(train, batch_size=batch_size, shuffle=True)

#model
model = Classify()
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#training
print('\n Classification real iris data')
for epoch in range(max_epoch):
  for i, (x,y) in enumerate(train_ld):
    
    #forward
    optimizer.zero_grad()
    out = model(x)
    model_loss = loss(out,y)

    #backward
    model_loss.backward()
    optimizer.step()

  if epoch%5 == 0:
    print(f'epoch: {epoch}/{max_epoch}, loss: {model_loss:.4f}')
    

#accracy test
test_ld = DataLoader(test, batch_size = 1)
model.eval()
with torch.no_grad():
  acc = 0
  for i, (x,y) in enumerate(test_ld):
    out = model(x)
    
    out = torch.round(out)
    if (out == y).all():
      acc +=1

  print(f'accuracy is {acc}/{len(test_ld)} --> {acc/len(test_ld)*100:.2f}%')
  
  
  
#fake_Dataset

#prepare data
fake_data = pd.read_csv('rescaledfake_iris.csv')
data = my_dataset(data=fake_data)
data.y = torch.nn.functional.one_hot(data.y) 
data.y = data.y.type(torch.float32)
    
# hyperparams
batch_size = 10
lr = 0.005
max_epoch = 50


# loader
train, test = train_test_split(data)
train_ld = DataLoader(train, batch_size=batch_size, shuffle=True)

#model
model = Classify()
loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('\n Classification fake iris data')
#training
for epoch in range(max_epoch):
  for i, (x,y) in enumerate(train_ld):
    
    #forward
    optimizer.zero_grad()
    out = model(x)
    model_loss = loss(out,y)

    #backward
    model_loss.backward()
    optimizer.step()

  if epoch%5 == 0:
    print(f'epoch: {epoch}/{max_epoch}, loss: {model_loss:.4f}')
    
#accracy test
test_ld = DataLoader(test, batch_size = 1)
model.eval()
with torch.no_grad():
  acc = 0
  for i, (x,y) in enumerate(test_ld):
    out = model(x)
    
    out = torch.round(out)
    if (out == y).all():
      acc +=1

  print(f'accuracy is {acc}/{len(test_ld)} --> {acc/len(test_ld)*100:.2f}%')