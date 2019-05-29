from torch import Tensor, softmax
from torchvision.models import vgg16
img_sizes = (3,224,224)

premodel = vgg16(pretrained=True).eval()


with open('class_keys.txt', 'r') as f:
    class_ids = tuple(line for line in f.read().split("\n"))
with open('class_descriptions.txt', 'r') as f:
    class_descriptions = {}
    for line in f.read().split("\n"):
        line = line.split(" ")
        class_descriptions[line[0]] = line[1:]


def get_description(predicted_class):
    return class_descriptions[class_ids[predicted_class]]


def scores(inputs, wrt_class):
    return softmax(premodel(inputs), 1)[:, wrt_class]


