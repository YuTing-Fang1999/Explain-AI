# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import My_Model
import json

model =  My_Model(32, 2)

architecture = []
links = []
i=0
# Display all model layer weights
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    architecture.append(para.shape[0])
    for t in range(para.shape[0]):
        for s in range(para.shape[1]):
            source = "{}_{}".format(i, s)
            target = "{}_{}".format(i+1, t)
            id = "{}-{}".format(source, target)
            weight = para[t][s].tolist()

            link = {"id":id, "source":source, "target":target, "weight":weight}
            print(link)
            links.append(link)
    i+=1

    

print(architecture)

 
# Writing to sample.json
with open("links.json", "w") as outfile:
    outfile.write(json.dumps(links, indent=4))