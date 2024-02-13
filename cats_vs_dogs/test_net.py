import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.utils.data.dataloader as DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
if(torch.cuda.is_available()):
    device = torch.device("cuda")

transform_imgs = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

imgs_data = ImageFolder("./test_set", transform=transform_imgs)
testing_loader = DataLoader(imgs_data, batch_size=1, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64* 28 * 28, 500)
        self.fc2 = nn.Linear(500, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))
        return x

model = Net().to(device)
model.load_state_dict(torch.load("myLittleModel.pth"))
for img, labels in testing_loader:
    output = model(img)
    print(output)