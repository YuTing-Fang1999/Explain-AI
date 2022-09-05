import json 
import wandb

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        # y = np.tanh(self.y[idx])
        return x, y

    def __len__(self):
        return len(self.y)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model =  My_Model(32, 2).to(device)

wandb.init(
      # Set the project where this run will be logged
      project="ML2022Spring_HW3",
      name= '{} bat={} L={} lr={} wd={} {}'.format(myseed, batch_size, n_layer, lr_rate, weight_decay, _exp_name)
    )
    wandb.watch(model)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
epoch_n = 300
bs = 16

train_dataset = My_Dataset(data["x_train"], data["y_train"])
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
model.train()
loss_record = []
loss_plot = []

for epoch in range(epoch_n):
    for x, y in train_loader:
        x = x.to(device)
        output = model(x)
        loss = criterion(output, y)
        # Compute gradient(backpropagation).
        loss.backward()
        # Update parameters.
        optimizer.step()
        loss_record.append(loss.detach().item())
    
    if (epoch+1) % 10 == 0:
        mean_train_loss = sum(loss_record)/len(loss_record)
        loss_plot.append(mean_train_loss)  # plot loss

plt.plot(range(len(loss_plot)), loss_plot)
plt.show()