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
    transforms.Resize(255), #(255, 255) ???
    transforms.CenterCrop(224), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
datasets_folder = datasets.ImageFolder("../cifar_10_network/test/", transform=transform_imgs)

img_labels = torch.Tensor(len(datasets_folder))
img_data = torch.Tensor(len(datasets_folder), 3, 255, 255)
for i in range(len(datasets_folder)):
    img_labels[i] = (datasets_folder[i][1])
for i in range(len(datasets_folder)):
    for j in range(3):
        img_data[i][j] = datasets_folder[i][0][j]
x_train, x_val, y_train, y_val = train_test_split(img_data, img_labels, shuffle=True, random_state=2024, test_size=0.2)

train_loader = torch.utils.data.DataLoader((x_train, y_train), batch_size=64)
val_loader = torch.utils.data.DataLoader((x_val, y_val), batch_size=64)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_unit_1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_unit_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_unit_4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(14*14*128, 128)
        self.fc2 = nn.Linear(128, 1)
        self.final = nn.Sigmoid()
    def forward(self, x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = self.conv_unit_3(out)
        out = self.conv_unit_4(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.final(out)
        return out
def evaluate(model, data_loader):
    loss = []
    correct = 0.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            model.eval()
            output = model(images)
            predicted = output>0.5
            correct += (labels.reshape(-1,1) == predicted.reshape(-1, 1)).float().sum()
            del([images, labels])
            if device == "cuda":
                torch.cuda.empty_cache()
    print('\nVal Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(data_loader.dataset), 100. * correct/len(data_loader.dataset)))
model = CNN()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.BCELoss()
total_step = len(train_loader)
print("Total Batches: ",total_step)
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs.float(), images.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    evaluate(model, val_loader)
        
#print("----------------")

