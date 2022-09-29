import json
from select import select 
import wandb 
import numpy as np 

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import My_Model

with open('dataset/dataset.json') as f:
    data = json.load(f)

select_threshold = 0
# clean data
x_train = []
y_train = []
data["y_train"] = np.array(data["y_train"])
for i in range(len(data["y_train"])):
    if (np.abs(data["y_train"][i]) >=select_threshold).any():
        x_train.append(data["x_train"][i])
        y_train.append(list(data["y_train"][i]))

pow = 0.9
zero_threshold = 0
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
        # print(y)
        sign = y<0
        y = np.around(np.abs(y), 5)
        y[y<zero_threshold] = 0

        y = np.power(y, pow)
        y[sign] = -y[sign]
        # print(y)
        
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
bs = 16
lr_rate = 1e-5
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

if log:
    config = dict (
        pow = pow,
        learning_rate = lr_rate,
        batch_size = bs,
        epoch = epoch_n,
        select_threshold = select_threshold,
        zero_threshold = zero_threshold,
        model = 'narrow'
        )

    wandb.init(
        # Set the project where this run will be logged
        project="Explain AI",
        name= 'pow={} bat={} lr={} epo={} s={} z={}'.format(pow, bs, lr_rate, epoch_n, select_threshold, zero_threshold),
        config=config
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
for i in range(30,len(y_train)):
    dataset = My_Dataset(x_train[:i], y_train[:i])
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
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(x_train[i])).detach().numpy()
            pred[np.abs(pred)<zero_threshold] = 0

            b = pred * y_train[i] >= 0
            print(x_train[i][-7:])
            print(y_train[i])
            print(pred)
            print(b.astype(int))
            
            acc.append(b.astype(int))
            print(np.array(acc).mean())
            print()
            if log: wandb.log({'acc': np.array(acc).mean()})
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
print(model) 

if log: wandb.finish()

torch.save(model.state_dict(), "My_Model")