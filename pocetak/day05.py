import torch
import torch.nn as nn

model = nn.Linear(1, 1)

print(model)
print(model.weight)
print(model.bias)