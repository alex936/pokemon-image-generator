"""Quick comparison script to visually compare two fitness functions."""
import os
import random

from PIL import Image
from tqdm import tqdm

from fitness import FITNESS
from individual import Individual
from make_gif import make_gif
from pokemon import get_pokemons
from settings import TARGET_IMAGES_DIR, BASE_DIR

POPULATION_SIZE = 4
MUTATION_RATE = 0.99
CROSSOVER_RATE = 0.3
POKEMON_SIZE = 15
NUM_GENERATIONS = 9000
SNAPSHOT_INTERVAL = 900
TARGET = "weepinbell.jpg"
FITNESS_FUNCTIONS = ["LABMSEFitness", "CIEDE2000Fitness", "VGGPerceptualFitness"]


def run(fitness_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    target_image = Image.open(os.path.join(TARGET_IMAGES_DIR, TARGET)).convert("RGB")
    Individual.target_image = target_image
    Individual.pokemons = get_pokemons(POKEMON_SIZE)
    Individual.max_num_pokemons = NUM_GENERATIONS

    population = [Individual.get_random_individual() for _ in range(POPULATION_SIZE)]
    fitness_evaluator = FITNESS[fitness_name](target_image)

    for i in tqdm(range(NUM_GENERATIONS), desc=fitness_name):
        fitness_evaluator.get_fitness(population)
        ordered = sorted(population, key=lambda ind: ind.fitness)
        parents = ordered[-2:]

        if i % SNAPSHOT_INTERVAL == 0:
            best = parents[-1]
            best.genotype.save(os.path.join(output_dir, f"best_ind_iter_{i:06d}.png"))

        new_population = list(parents)
        for _ in range(POPULATION_SIZE):
            individual = random.choice(parents).copy()
            if random.random() < CROSSOVER_RATE:
                individual.apply_crossover(random.choice(parents))
            if random.random() < MUTATION_RATE:
                individual.apply_mutation()
            new_population.append(individual)
        population = new_population

    best = sorted(population, key=lambda ind: ind.fitness)[-1]
    best.genotype.save(os.path.join(output_dir, f"best_ind_iter_{NUM_GENERATIONS:06d}.png"))

    make_gif(output_dir)
    print(f"Saved {fitness_name} GIF to {output_dir}")


if __name__ == "__main__":
    for fitness_name in FITNESS_FUNCTIONS:
        run(fitness_name, os.path.join(BASE_DIR, "compare", fitness_name))
