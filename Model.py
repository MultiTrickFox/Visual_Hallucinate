from torch import Tensor, softmax, pow, no_grad, set_grad_enabled
from torchvision.models import vgg16
img_sizes = (3,224,224)

premodel = vgg16(pretrained=True).eval()


with open('class_keys.txt', 'r') as f:
    class_ids = tuple(line for line in f.read().split("\n") if line != "")
with open('class_descriptions.txt', 'r') as f:
    class_descriptions = {}
    for line in f.read().split("\n"):
        line = line.split(" ")
        class_descriptions[line[0]] = line[1:]

hm_classes = len(class_ids)

def get_description(predicted_class):
    return class_descriptions[class_ids[predicted_class]]

def scores(inputs, wrt_class):
    return softmax(premodel(inputs), 1)[:, wrt_class]


def train_population(population, wrt_class):
    label = Tensor([[0 if i != wrt_class else 1 for i in range(hm_classes)] for _ in range(len(population))])
    loss = pow(label - softmax(premodel(population), 1), 2)
    loss.sum().backward()
    with no_grad():
        if population.grad != None:
            population -= population.grad()
        else: print(f'no grad. {population.requires_grad}')
    population.grad = None
    return population
