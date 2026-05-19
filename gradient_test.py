import torch
import torch.nn as nn

# 3 layer nn
layer1 = nn.Linear(8,8)
layer2 = nn.Linear(8,8)
layer3 = nn.Linear(8,2)

# 16 samples with 8 features each
x = torch.randn(16,8)
labels = torch.randint(0,2,(16,)) # what does this lien even mean??

out = layer1(x)
out = layer2(out)
out = torch.relu(out)
out = layer3(out)

loss = nn.CrossEntropyLoss()(out, labels)

loss.backward()

print("=== GRADIENT NORMS PER LAYER ===")
print(f"layer1: {layer1.weight.grad.norm().item():.6f}")
print(f"layer2: {layer2.weight.grad.norm().item():.6f}")
print(f"layer3: {layer3.weight.grad.norm().item():.6f}")

for layer in [layer1, layer2, layer3]:
    layer.zero_grad()