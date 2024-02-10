import PIL.Image as Image
import pandas
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
datasets_folder = datasets.ImageFolder("../cifar_10_network/test/")
data = train_test_split(datasets_folder, test_size=0.2)

img = Image.fromarray(data[0][0][0])
img.show()