import winsound
duration = 500
freq = 440

import pandas as pd
import numpy as np
import seaborn as sns
import torch
from gan import Discriminator, Generator
from my_iris_dataset import my_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

filename = ''
#hyperparams
batch_size = 10
lr = 0.0003
max_epoch = 600


#prepare data
iris = sns.load_dataset('iris')
mapp = {'setosa' : 0, 'versicolor' : 1, 'virginica': 2}
iris.species = iris.species.apply(lambda x : mapp[x])
data = my_dataset(iris,transform=True)
train_ld = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

#real species means
mean0 = data.iris[data.iris.species == 0].mean().values[:-1]
mean1 = data.iris[data.iris.species == 1].mean().values[:-1]
mean2 = data.iris[data.iris.species == 2].mean().values[:-1]
err = [[],[],[]]

#model
dis = Discriminator(hidden_nodes= 10)
gen = Generator(noise_dim = 6, hidden_nodes = 10)

#set mode
dis.train()            
gen.train()

#optimizers and loss functions
dis_optimizer = torch.optim.Adam(dis.parameters(), lr, betas=(0.9,0.999))
gen_optimizer = torch.optim.Adam(gen.parameters(), lr, betas=(0.5,0.999))
loss = torch.nn.BCELoss(reduction = 'mean')
all_ones = torch.ones(batch_size, dtype=torch.float32)
all_zeros = torch.zeros(batch_size, dtype=torch.float32)


print("\nStarting training ")

for epoch in range(max_epoch):
    for batch_idx, (x,y) in enumerate(train_ld):

        # train discriminator 
        dis_optimizer.zero_grad() #azzera il gradiente
        dis_real_oupt = dis(x,y).view(-1) #output messo in un vettore riga
        dis_real_loss = loss(dis_real_oupt, all_ones)  #a real data point has to be all_ones

        # train discriminator using fake datapoint
        noise = torch.normal(0.0, 1.0, size=(batch_size,gen.noise_dim)) 
        fake_data = gen(noise,y)  #fake data point , labeled with the same target as real images

        dis_fake_oupt = dis(fake_data,y).view(-1)
        dis_fake_loss = loss(dis_fake_oupt, all_zeros)    # fake data has to be all_zeros 

        dis_total_loss = dis_real_loss + dis_fake_loss

        dis_total_loss.backward()  # compute gradients
        dis_optimizer.step()     # update weights and biases
        
    for batch_idx, (x,y) in enumerate(train_ld):
        #train generator with fake data
        gen_optimizer.zero_grad()
        noise = torch.normal(0.0, 1.0, size= (batch_size, gen.noise_dim))
        fake_data = gen(noise,y) #fake data, labeled as y, forced to be true
        dis_fake_oupt = dis(fake_data,y).view(-1)
        gen_loss = loss(dis_fake_oupt, all_ones)

        gen_loss.backward()
        gen_optimizer.step()
    
    #validation: distance from mean
    with torch.no_grad():
        for i,m in enumerate([mean0,mean1,mean2]):
            noise = torch.normal(0.0, 1.0, size=(50, gen.noise_dim))
            y = torch.ones(50, dtype=torch.int64)*i
            fake_data = gen(noise,y)
            fake_db = pd.DataFrame(np.array(fake_data), columns=data.iris.columns[:-1])
            err[i].append(np.linalg.norm(fake_db.mean().values-m))
            
    if epoch % 30 == 0:
          print(f'epoch {epoch:2d} of {max_epoch} - dis loss : {dis_total_loss:.3f} - gen loss : {gen_loss:.3f}' )

print("Training complete ")


#plot fake data
with torch.no_grad():
    for i in range(3):
        noise = torch.normal(0.0, 1.0, size=(50, gen.noise_dim))
        y = torch.ones(50, dtype=torch.int64)*i
        fake_data = gen(noise,y)
        fake_db = pd.DataFrame(np.array(fake_data), columns=data.iris.columns[:-1])
        fake_db['species'] = 3
        new_df = pd.concat([data.iris, fake_db])
    
        pg = sns.PairGrid(new_df,  palette='Set1', hue = 'species', x_vars = ['sepal_length','sepal_width'], y_vars = ['petal_length','petal_width'])
        pg.map(sns.scatterplot)
        pg.fig.suptitle(f'Generating species {i}')
        pg.add_legend(title = 'species')
        pg.fig.savefig(f'{filename}_species{i}.png')

#generate fake dataset
with torch.no_grad():
    noise = torch.normal(0.0, 1.0, size=(150, gen.noise_dim))
    y = torch.tensor([0]*50 + [1]*50 + [2]*50 , dtype=torch.int64)
    fake_data = gen(noise,y)
    fake_data = data.inverse_scaling(fake_data)
    fake_iris = pd.DataFrame(np.array(fake_data), columns = data.iris.columns[:-1])
    fake_iris['species'] = y
    fake_iris.to_csv(filename+'fake_iris.csv', index = False)

#plot distance from meand
fig, ax =plt.subplots()
ax.plot(err[0], label = 'species 0')
ax.plot(err[1], label = 'species 1')
ax.plot(err[2], label = 'species 2')
ax.set_xlabel('iteration')
ax.set_ylabel('euclidian distance from mean')
plt.savefig(filename+'distance_from_mean.png')
plt.legend()
plt.show()

winsound.Beep(freq, duration)