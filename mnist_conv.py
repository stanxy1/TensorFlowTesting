import torch
import torchvision
import pandas 
import numpy
from sklearn.model_selection import train_test_split

data_from_csv = torch.Tensor(pandas.read_csv("./mnist/mnist_test.csv").to_numpy())
#indices = torch.tensor([0,1])
labels_csv = data_from_csv[:, :1]
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
