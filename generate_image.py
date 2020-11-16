import argparse
import os
import random

from PIL import Image
from tqdm import tqdm

from fitness import FITNESS
from individual import Individual
from make_gif import make_gif
from pokemon import get_pokemons
from settings import PLOT_DIR, TARGET_IMAGES_DIR

POPULATION_SIZE = 4
MUTATION_RATE = 0.99
CROSSOVER_RATE = 0.3

HEIGHT = 300
WIDTH = 300
POKEMON_SIZE = 15

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--target",
        dest="target",
        type=str,
        help="Filename of target image. Located in data/target_images/",
        required=False,
        default="weepinbell.jpg",
    )

    arg_parser.add_argument(
        "--fitness",
        dest="fitness",
        type=str,
        choices=FITNESS.keys(),
        help="Choose type of fitness metric.",
        required=False,
        default="LABMSEFitness",
    )

    arg_parser.add_argument(
        "-g",
        "--num-generations",
        dest="num_generations",
        type=int,
        required=False,
        default=80000,
    )

    args = arg_parser.parse_args()

    target_image = Image.open(os.path.join(TARGET_IMAGES_DIR, args.target))
    target_image = target_image.convert("RGB")

    Individual.target_image = target_image
    Individual.pokemons = get_pokemons(POKEMON_SIZE)
    Individual.max_num_pokemons = args.num_generations

    population = [Individual.get_random_individual() for _ in range(POPULATION_SIZE)]
    fitness_class = FITNESS[args.fitness]
    fitness_evaluator = fitness_class(Individual.target_image)

    for i in tqdm(range(args.num_generations)):
        fitness_evaluator.get_fitness(population)
        ordered_individuals = sorted(population, key=lambda i: i.fitness)
        fittest_individual = ordered_individuals[-1]
        # Parent selection: Select the 2 best individuals
        parents = ordered_individuals[-2:]

        if i % 1000 == 0:
            print("Saving best individual after generation {}".format(i))

            fittest_individual.genotype.save(
                os.path.join(
                    PLOT_DIR,
                    "best_ind_iter_{}.png".format(i),
                )
            )

        new_population = []
        new_population.extend(parents)

        for i in range(POPULATION_SIZE):
            random_parents = random.sample(parents, k=2)
            individual = random_parents[0].copy()
            # crossover
            if random.random() < CROSSOVER_RATE:
                individual.apply_crossover(individual)
            # mutate
            if random.random() < MUTATION_RATE:
                individual.apply_mutation()

            new_population.append(individual)
        population = new_population

    make_gif(PLOT_DIR)
