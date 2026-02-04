import torch
state = torch.load("best_model_fixed.pth", map_location="cpu")
print(state.keys())
