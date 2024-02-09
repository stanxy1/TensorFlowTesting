import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.cuda as cuda

if cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
transform_imgs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(255),
    transforms.CenterCrop(244),
    transforms.Normalize()
])