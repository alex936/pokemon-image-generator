import random

from PIL import Image

class Individual:
    """Represents a single candidate solution in the genetic algorithm."""

    pokemons = None
    target_image = None

    def __init__(self, genotype, fitness=None):
        """

        :param genotype: PIL Image representing the individual's composition of Pokemon tiles
        :param fitness: Fitness score indicating similarity to the target image
        """
        self.genotype = genotype
        self.fitness = fitness

    def apply_mutation(self):
        """Mutate the individual by pasting a random Pokemon at a random position."""
        pokemon = random.choice(self.pokemons)
        x = random.randint(0, Individual.target_image.width)
        y = random.randint(0, Individual.target_image.height)
        self.genotype.paste(pokemon, box=(x, y), mask=pokemon)

    def apply_crossover(self, other_individual):
        """Apply crossover by copying the top-left region from another individual.

        :param other_individual: The other Individual to copy a region from
        """
        height_cutoff = random.randint(1, Individual.target_image.size[1])
        box_top = (0, 0, int(self.genotype.size[0] / 2), height_cutoff)
        other_individual_cropped = other_individual.genotype.crop(box_top)
        self.genotype.paste(other_individual_cropped, box_top)

    def set_fitness(self, fitness):
        """Set the fitness score for this individual.

        :param fitness: Fitness score as a float
        """
        self.fitness = fitness

    @staticmethod
    def get_random_individual():
        """Create a new individual with a blank white genotype image."""
        image = Image.new(mode=Individual.target_image.mode, size=Individual.target_image.size, color=(255, 255, 255))
        return Individual(genotype=image)

    def copy(self):
        """Return a deep copy of this individual."""
        return Individual(genotype=self.genotype.copy(), fitness=self.fitness)
