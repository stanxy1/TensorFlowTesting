import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize([28, 28]), torchvision.transforms.Grayscale(num_output_channels=1)])
test_imgs = torchvision.datasets.ImageFolder("./mnist_test",transform=transforms)

test_loader = torch.utils.data.DataLoader(test_imgs, batch_size=64)

class Conv_Network(nn.Module):
    def __init__(self):
        super(Conv_Network, self).__init__()
        self.conv_unit_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

model = Conv_Network()
model.load_state_dict(torch.load("mnist_conv.pth"))
pos_classes = ['1','2','3','4','5','6','7','8','9']
for img_data,labels in test_loader:
     
    output = model(img_data)
    for t in output:
        index = (t.detach().numpy().argmax())-1
        print(pos_classes[index])
    
