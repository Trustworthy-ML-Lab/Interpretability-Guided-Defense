# import timm
# print(timm.__version__)

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5, 3),
    nn.Linear(3, 2),
)

x = torch.randn((1, 2))

y = model(x)
print('direct forward pass through Sequential')
print(y)

def model_loop(model, x):
    for i, layer in enumerate(model):
        x = layer(x)
        print(i)
    return x

y = model_loop(model, x)
print('looped forward pass over Sequential')
print(y)