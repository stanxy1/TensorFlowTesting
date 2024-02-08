import torch
import torchvision
import pandas 
import numpy
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
data_from_csv = torch.Tensor(pandas.read_csv("./mnist/mnist_test.csv").to_numpy())
#indices = torch.tensor([0,1])
labels_csv = data_from_csv[:, :1].long()
train_test_data_csv = data_from_csv[:, 1:]
labels_csv = labels_csv.flatten()
x_train, x_test, y_train, y_test = train_test_split(train_test_data_csv, labels_csv, shuffle=True, random_state=2024, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28) 
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

x_train = x_train/255.0
x_test = x_test/255.0
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

class Conv_Network(nn.Module):
    def __init__(self):
        super(Conv_Network, self).__init__()
        self.conv_unit_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_unit_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(7*7*32, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        out = self.conv_unit_1(x)
        out = self.conv_unit_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out
def evaluate(data_loader):
    model.eval()
    loss = 0
    with torch.no_grad():
        for data, label in data_loader:
            output = model(data)
            loss += F.cross_entropy(output, label, size_average=True).data.item()
        
    print("Validation set loss is %.4f" % loss)
model = Conv_Network()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
num_epochs = 10
for epoch in range(num_epochs):
    for data, labels in train_loader:
        output = model(data)

        loss = loss_func(output, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: {}, the most recent training loss: {}".format(epoch,loss))
    evaluate(test_loader)
torch.save(model.state_dict(), "mnist_conv.pth")

        
