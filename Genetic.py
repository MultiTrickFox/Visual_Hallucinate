from Model import scores, img_sizes
from torch import randn, stack
from random import random


def create_population(hm):
    return randn(hm, *img_sizes)


def mutate_population(population, mutation_prob, mutation_effect):
    hm_cols = img_sizes[-1]
    # TODO : parallelize here.
    for thing in population:
        for channel in thing:
            for row in channel:
                for col_nr in range(hm_cols):
                    if random() < mutation_prob:
                        row[col_nr] += mutation_effect*random()
    # return population


# crossover (population) => a += crossover_effect * b (if randn() < crossover_prob)


def mostfits(population, hm, wrt_class):
    sc = scores(population, wrt_class)
    mostfits = []
    for i in range(hm):
        fit = sc.argmax()
        mostfits.append(population[fit])
        sc[fit] = 0
    return stack(mostfits, 0)
