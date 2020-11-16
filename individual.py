import random

from PIL import Image


class Individual:
    pokemons = None
    target_image = None

    def __init__(self, genotype, fitness=None):
        self.genotype = genotype
        self.fitness = fitness

    def apply_mutation(self):
        pokemon = random.choice(self.pokemons)
        x = random.randint(0, Individual.target_image.width)
        y = random.randint(0, Individual.target_image.height)
        self.genotype.paste(pokemon, box=(x, y), mask=pokemon)

    def apply_crossover(self, other_individual):
        height_cutoff = random.randint(1, Individual.target_image.size[1])
        box_top = (0, 0, int(self.genotype.size[0] / 2), height_cutoff)
        other_individual_cropped = other_individual.genotype.crop(box_top)
        self.genotype.paste(other_individual_cropped, box_top)

    def set_fitness(self, fitness):
        self.fitness = fitness

    @staticmethod
    def get_random_individual():
        image = Image.new(mode=Individual.target_image.mode, size=Individual.target_image.size, color=(255, 255, 255))
        return Individual(genotype=image)

    def copy(self):
        return Individual(genotype=self.genotype.copy(), fitness=self.fitness)
