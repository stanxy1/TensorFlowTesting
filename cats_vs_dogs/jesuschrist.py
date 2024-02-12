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
    transforms.Resize((255,255)), #(255, 255) ???
    transforms.CenterCrop((224,224)), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_img = datasets.ImageFolder("./archive/train", transform=transform_imgs)
val_img = datasets.ImageFolder("./archive/test", transform=transform_imgs)

train_loader = torch.utils.data.DataLoader(test_img, batch_size=32, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_img, batch_size=32, pin_memory=True)

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Sequential(
    nn.Linear(model.classifier[6].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1),
    nn.Sigmoid())
num_epochs = 10
loss_func = nn.BCELoss()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Total step: ", len(train_loader))
def evaluate(model, data_loader):
    #loss = []
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

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs.float(), labels.float().view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    evaluate(model, val_loader)