from Genetic import create_population, mutate_population, mostfits
from Model import train_population

from torch import cat
from PIL import Image
from pickle import dump, load


hm_iterations = 1_000

hm_initial_population = 10#100#500

crossover_chance = 0.2
crossover_effect = 0.5

mutation_chance = 0.4
mutation_effect = 0.1

hm_mostfits = 3#50#500

target_class = 47



population = create_population(hm_initial_population)

for _ in range(hm_iterations):
    print(f'Iterating: {_}')

    population = cat([population.clone(), mutate_population(population.clone(), mutation_chance, mutation_effect)], 0)
    print(flush=True)
    population = mostfits(population, hm_mostfits, target_class)
    print(flush=True)
    # population = cat([population.clone(), train_population(population.clone(), target_class)], 0)
    # population = mostfits(population, hm_mostfits, target_class)
    print(flush=True)

    if _%10==0:
        Image.fromarray(population[0].detach().numpy().swapaxes(0,-1), 'RGB').show()


dump(population, 'population.p')
dump(population[0], 'fittest.p')


# mutates = mutate(population)

# crossovers = crossover(population)

# mutated_crossovers = mutate(cross_overs)

# crossovered_mutates = crossover(mutates)

# population = population + crossovers + mutates + crossovered_mutates + mutated_crossovers
