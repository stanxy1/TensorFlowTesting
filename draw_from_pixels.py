import PIL.Image as Image
import pandas
values = pandas.read_csv("./mnist/mnist_test.csv").to_numpy()

img = Image.fromarray(values[:, 1:])
img.show()