import torch
from torch import nn
import torchvision
import torch.onnx

# An instance of your model


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mnasnet1_0(num_classes=10)
        state_dict = torch.load("ava_cnn.pt")
        self.model.load_state_dict(state_dict)
    
    def forward(self, x):
        return self.model(x).softmax(-1)

# An example input you would normally provide to your model's forward() method
x = torch.zeros(1, 3, 224, 224)
model = Model()
model.train(False)

print(model(x))

# model = torchvision.models.mnasnet1_0(num_classes=10)
# state_dict = torch.load("ava_cnn.pt")
# model.load_state_dict(state_dict)

# Export the model
torch_out = torch.onnx._export(model, x, "mnasnet1_0.onnx")
