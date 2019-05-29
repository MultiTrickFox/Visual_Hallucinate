from Model import scores, img_sizes
from torch import randn, stack

from random import random
from joblib import delayed, Parallel ; hm_cpu = 4


def create_population(hm):
    return randn(hm, *img_sizes, requires_grad=True)


def mutate_population(population, mutation_prob, mutation_effect):
    print('\tmutating: ',end='',flush=True)
    @delayed
    def mutate_sample(sample, mutation_prob, mutation_effect):
        hm_cols = img_sizes[-1]
        for channel in sample:
                for row in channel:
                    for col_nr in range(hm_cols):
                        if random() < mutation_prob:
                            row[col_nr] += mutation_effect*random()
        print('/',end='',flush=True)
        return sample
    with Parallel(hm_cpu) as p:
        return stack(p(mutate_sample(thing, mutation_prob, mutation_effect) for thing in population), 0)



# crossover (population) => a += crossover_effect * b (if randn() < crossover_prob)


def mostfits(population, hm, wrt_class):
    print('\tselecting: ',end='',flush=True)
    sc = scores(population, wrt_class)
    mostfits = []
    for i in range(hm):
        fit = sc.argmax()
        mostfits.append(population[fit])
        sc[fit] = 0
        print('/',end='',flush=True)
    return stack(mostfits, 0)
