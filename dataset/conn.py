import json
import numpy as np

with open('dataset1.json') as f1:
    data1 = json.load(f1)

with open('dataset2.json') as f2:
    data2 = json.load(f2)

data = {}
data["x_train"] = np.concatenate([data1["x_train"], data2["x_train"]])
data["y_train"] = np.concatenate([data1["y_train"], data2["y_train"]])

with open("dataset.json", "w") as outfile:
    data["x_train"] = data["x_train"].tolist()
    data["y_train"] = data["y_train"].tolist()
    json.dump(data, outfile)
