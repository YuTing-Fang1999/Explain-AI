# Pytorch
import torch
import torch.nn as nn
import numpy as np
import json

class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            # nn.Linear(input_dim, 16, bias=False),
            # nn.ReLU(),
            # nn.Linear(16, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, output_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            # nn.Linear(input_dim, input_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(32, 16, bias=False),
            # nn.ReLU(),
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

model = My_Model(32, 2)
model.load_state_dict(torch.load("My_Model"))
model.eval()

# for i in range(32):
#     x = np.zeros(32)
#     x[i] = 1
#     print(model(torch.FloatTensor(x)))

x = np.zeros(32)
# x[30] = 1
print(model(torch.FloatTensor(x)))

# x = np.zeros(32)
# x[31] = 1
# print(model(torch.FloatTensor(x)))

# x = np.zeros(32)
# x[30] = 1
# x[31] = 1
# print(model(torch.FloatTensor(x)))

with open('dataset/dataset.json') as f:
    data = json.load(f)

x_train = data["x_train"]
y_train = data["y_train"]

acc = []
for i in range(len(y_train)):
    pred = model(torch.FloatTensor(x_train[i])).detach().numpy()
    b = pred * y_train[i] >=0
    # print(b)
    acc.append(b.astype(int))
    # print(x_train[i][-7:])
    # print(y_train[i])
    # print(model(torch.FloatTensor(x_train[i])))
    # print()
print(np.array(acc).mean())

# criterion = nn.MSELoss(reduction='mean')
# print(criterion(torch.FloatTensor([0]),torch.FloatTensor([1])))
# print(criterion(torch.FloatTensor([1]),torch.FloatTensor([0])))
# print(criterion(torch.FloatTensor([0]),torch.FloatTensor([0])))
# print(criterion(torch.FloatTensor([1]),torch.FloatTensor([1])))

