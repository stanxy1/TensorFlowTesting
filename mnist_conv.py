import torch
import torchvision
import pandas 
import numpy
from sklearn.model_selection import train_test_split

data_from_csv = torch.Tensor(pandas.read_csv("./mnist/mnist_test.csv").to_numpy())
#indices = torch.tensor([0,1])
labels_csv = data_from_csv[:, :1]
train_test_data_csv = data_from_csv[:, 1:]
x_train, x_test = train_test_split(train_test_data_csv, shuffle=False, test_size=0.2)
y_train, y_test = train_test_split(labels_csv, shuffle=False, test_size=0.2)
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)

print(x_train.shape)