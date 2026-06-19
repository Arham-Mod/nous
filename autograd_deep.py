import torch
import torch.nn as nn

x = torch.randn(4, 4, requires_grad = True)
y = x * 2
z = y.sum()
z.backward()

print(x.grad)

model = nn.Linear(4, 4)

x = torch.randn(8,4)

out = model(x)
print(out)