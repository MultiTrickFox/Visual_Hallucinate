from PIL import Image
from Genetic import create_population, mutate_population, mostfits


hm_iterations = 200

hm_initial_population = 500

crossover_chance = 0.2
crossover_effect = 0.5

mutation_chance = 0.4
mutation_effect = 0.1

hm_mostfits = 500

target_class = 47



population = create_population(hm_initial_population)

for _ in range(hm_iterations):

    population += mutate_population(population, mutation_chance, mutation_effect)
    population = mostfits(population, hm_mostfits, target_class)

    if _%10==0:
        Image.fromarray(population[0].numpy().swapaxes(0,-1), 'RGB').show()


# mutates = mutate(population)

# crossovers = crossover(population)

# mutated_crossovers = mutate(cross_overs)

# crossovered_mutates = crossover(mutates)

# population = population + crossovers + mutates + crossovered_mutates + mutated_crossovers
