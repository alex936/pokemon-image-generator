import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.measure import compare_ssim


def calculate_mse(image_a, image_b):
    err = np.sum((image_a - image_b) ** 2)
    return err


def calculate_mae(image_a, image_b):
    err = np.sum(np.abs(image_a - image_b))
    return err


class RGBMSEFitness():

    def __init__(self, target_image_pil):
        self.target_image_np = np.array(target_image_pil)

    def get_fitness(self, individuals):
        for individual in individuals:
            fitness_value = 1 / (
                    1
                    + calculate_mse(
                self.target_image_np, np.array(individual.genotype)
            )
            )
            individual.set_fitness(fitness_value)


class SSIMFitness():
    DOWNSCALED_SIZE = (100, 100)

    def __init__(self, target_image_pil):
        self.target_image_np = self.preprocess_pil_image(target_image_pil)

    @staticmethod
    def preprocess_pil_image(pil_image):
        return np.array(
            pil_image.resize(
                SSIMFitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
            )
        )

    def evaluate_fitness(self, individuals):
        for individual in individuals:
            fitness_value = compare_ssim(
                self.target_image_np,
                self.preprocess_pil_image(individual.phenotype),
                multichannel=True,
            )
            individual.set_fitness(fitness_value)


class LABMSEFitness():
    DOWNSCALED_SIZE = (100, 100)

    def __init__(self, target_image_pil):
        self.target_image_np_lab = self.preprocess_pil_image(target_image_pil)

    @staticmethod
    def preprocess_pil_image(pil_image):
        return rgb2lab(
            np.array(
                pil_image.resize(
                    LABMSEFitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
                )
            )
        )

    def get_fitness(self, individuals):
        for individual in individuals:
            fitness_value = 1 / (
                    1
                    + calculate_mse(
                self.target_image_np_lab,
                self.preprocess_pil_image(individual.genotype),
            )
            )
            individual.set_fitness(fitness_value)


FITNESS = {
    "RGBMSE": RGBMSEFitness,
    "SSIM": SSIMFitness,
    "LABMSEFitness": LABMSEFitness
}
