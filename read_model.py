# Pytorch
import torch
import torch.nn as nn
import numpy as np
import json
from model import My_Model


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
    pred[np.abs(pred)<0.05] = 0
    b = pred * y_train[i] >= 0
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

