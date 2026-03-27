import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

def calculate_mse(image_a, image_b):
    """Calculate the mean squared error between two images.

    :param image_a: First image as a numpy array
    :param image_b: Second image as a numpy array
    """
    err = np.sum((image_a - image_b) ** 2)
    return err


def calculate_mae(image_a, image_b):
    """Calculate the mean absolute error between two images.

    :param image_a: First image as a numpy array
    :param image_b: Second image as a numpy array
    """
    err = np.sum(np.abs(image_a - image_b))
    return err


class RGBMSEFitness():
    """Fitness evaluator using mean squared error in RGB color space."""

    def __init__(self, target_image_pil):
        """

        :param target_image_pil: Target image as a PIL Image in RGB mode
        """
        self.target_image_np = np.array(target_image_pil)

    def get_fitness(self, individuals):
        """Assign fitness scores to individuals based on RGB MSE against the target image.

        :param individuals: List of Individual objects to evaluate
        """
        for individual in individuals:
            fitness_value = 1 / (
                    1
                    + calculate_mse(
                self.target_image_np, np.array(individual.genotype)
            )
            )
            individual.set_fitness(fitness_value)


class SSIMFitness():
    """Fitness evaluator using the Structural Similarity Index (SSIM)."""

    DOWNSCALED_SIZE = (100, 100)

    def __init__(self, target_image_pil):
        """

        :param target_image_pil: Target image as a PIL Image
        """
        self.target_image_np = self.preprocess_pil_image(target_image_pil)

    @staticmethod
    def preprocess_pil_image(pil_image):
        """Downscale a PIL image and convert it to a numpy array.

        :param pil_image: Input PIL Image
        """
        return np.array(
            pil_image.resize(
                SSIMFitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
            )
        )

    def evaluate_fitness(self, individuals):
        """Assign fitness scores to individuals based on SSIM against the target image.

        :param individuals: List of Individual objects to evaluate
        """
        for individual in individuals:
            fitness_value = structural_similarity(
                self.target_image_np,
                self.preprocess_pil_image(individual.phenotype),
                channel_axis=-1,
            )
            individual.set_fitness(fitness_value)


class LABMSEFitness():
    """Fitness evaluator using mean squared error in the perceptual LAB color space."""

    DOWNSCALED_SIZE = (100, 100)

    def __init__(self, target_image_pil):
        """

        :param target_image_pil: Target image as a PIL Image in RGB mode
        """
        self.target_image_np_lab = self.preprocess_pil_image(target_image_pil)

    @staticmethod
    def preprocess_pil_image(pil_image):
        """Downscale a PIL image and convert it to a LAB color space numpy array.

        :param pil_image: Input PIL Image
        """
        return rgb2lab(
            np.array(
                pil_image.resize(
                    LABMSEFitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
                )
            )
        )

    def get_fitness(self, individuals):
        """Assign fitness scores to individuals based on LAB MSE against the target image.

        :param individuals: List of Individual objects to evaluate
        """
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
