import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
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
datasets_folder = datasets.ImageFolder("./PetImages", transform=transform_imgs)
