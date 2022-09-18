import json 
import wandb 
import numpy as np 

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

with open('dataset/dataset.json') as f:
    data = json.load(f)

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

pow = 0.7
class My_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

    def __init__(self, x, y):
        self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        y[y<0] = -np.power(-y[y<0], pow)
        y[y>0] = np.power(y[y>0], pow)
        # print(y)
        # .astype('float32')
        # y = np.tanh(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)

def train_valid_split(data_set, valid_ratio):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    return random_split(data_set, [train_set_size, valid_set_size])

log = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =  My_Model(32, 2).to(device)

epoch_n = 200
bs = 1
lr_rate = 1e-5
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

if log:
    wandb.init(
        # Set the project where this run will be logged
        project="Explain AI",
        name= 'pow={} bat={} lr={} epo={}'.format(pow, bs, lr_rate, epoch_n)
        )
    wandb.watch(model)

# dataset = My_Dataset(data["x_train"], data["y_train"])
# train_data, valid_data = train_valid_split(dataset, 0.3)
# print(train_data[:30][0].shape, train_data[:30][1].shape)
# print(train_data[:][0].shape, train_data[:][1].shape)
# print(len(train_data), len(valid_data))
# train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
# valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=True)

acc = []
for i in range(30,100):
    dataset = My_Dataset(data["x_train"][:i], data["y_train"][:i])
    train_loader = DataLoader(dataset, batch_size=bs, shuffle=True)

    model.train()
    loss_record = []
    for epoch in range(epoch_n):
        
        for x, y in train_loader:
            # print(x.shape, y.shape)
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            # Compute gradient(backpropagation).
            loss.backward()
            # Update parameters.
            optimizer.step()
            loss_record.append(loss.detach().item())
        
        # if (epoch+1) % 10 == 0:
    mean_train_loss = sum(loss_record)/len(loss_record)
    if log:
        wandb.log({'Train Loss': mean_train_loss, 'i':i-30})

    if i>50: #pred
        pred = model(torch.FloatTensor(data["x_train"][i])).detach().numpy()
        b = pred * data["y_train"][i] >=0
        acc.append(b.astype(int))
        # model.eval()
        # loss_record = []
        # for x, y in valid_loader:
        #     x, y = x.to(device), y.to(device)
        #     with torch.no_grad():
        #         pred = model(x)
        #         pred[pred>0] = 1
        #         pred[pred<0] = -1
        #         y[y>0]=1
        #         y[y<0]=-1
        #         loss = criterion(pred, y)

        #     loss_record.append(loss.item())
            
        # if (epoch+1) % 10 == 0:
        #     mean_valid_loss = sum(loss_record)/len(loss_record)
        #     if log:
        #         wandb.log({'Valid Loss': mean_valid_loss, 'epoch':epoch})
print(np.array(acc).mean())
if log:
    wandb.log({'acc': np.array(acc).mean()})
if log: wandb.finish()

torch.save(model.state_dict(), "My_Model")