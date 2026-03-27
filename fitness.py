import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
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
            if not individual.dirty:
                continue
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
            if not individual.dirty:
                continue
            fitness_value = structural_similarity(
                self.target_image_np,
                self.preprocess_pil_image(individual.phenotype),
                channel_axis=-1,
            )
            individual.set_fitness(fitness_value)


class LABMSEFitness():
    """Fitness evaluator using mean squared error in the perceptual LAB color space."""

    DOWNSCALED_SIZE = (100, 100)
    CACHE_KEY = "lab"

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
            if not individual.dirty:
                continue
            if self.CACHE_KEY not in individual._cache:
                individual._cache[self.CACHE_KEY] = self.preprocess_pil_image(individual.genotype)
            fitness_value = 1 / (1 + calculate_mse(self.target_image_np_lab, individual._cache[self.CACHE_KEY]))
            individual.set_fitness(fitness_value)


class LABSSIMFitness():
    """Fitness evaluator using SSIM in the perceptual LAB color space."""

    DOWNSCALED_SIZE = (100, 100)
    CACHE_KEY = "lab"

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
                    LABSSIMFitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
                )
            )
        )

    def get_fitness(self, individuals):
        """Assign fitness scores to individuals based on SSIM in LAB color space against the target image.

        :param individuals: List of Individual objects to evaluate
        """
        for individual in individuals:
            if not individual.dirty:
                continue
            if self.CACHE_KEY not in individual._cache:
                individual._cache[self.CACHE_KEY] = self.preprocess_pil_image(individual.genotype)
            fitness_value = structural_similarity(
                self.target_image_np_lab,
                individual._cache[self.CACHE_KEY],
                channel_axis=-1,
                data_range=self.target_image_np_lab.max() - self.target_image_np_lab.min(),
            )
            individual.set_fitness(fitness_value)


class CIEDE2000Fitness():
    """Fitness evaluator using the CIEDE2000 perceptual color difference metric."""

    DOWNSCALED_SIZE = (100, 100)
    CACHE_KEY = "lab"

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
                    CIEDE2000Fitness.DOWNSCALED_SIZE, resample=Image.BILINEAR
                )
            )
        )

    def get_fitness(self, individuals):
        """Assign fitness scores based on mean CIEDE2000 color difference against the target.

        :param individuals: List of Individual objects to evaluate
        """
        for individual in individuals:
            if not individual.dirty:
                continue
            if self.CACHE_KEY not in individual._cache:
                individual._cache[self.CACHE_KEY] = self.preprocess_pil_image(individual.genotype)
            delta = deltaE_ciede2000(self.target_image_np_lab, individual._cache[self.CACHE_KEY])
            individual.set_fitness(1 / (1 + delta.mean()))


class VGGPerceptualFitness():
    """Fitness evaluator using perceptual feature maps from a pretrained VGG16 network."""

    DOWNSCALED_SIZE = (64, 64)

    _transform = transforms.Compose([
        transforms.Resize(DOWNSCALED_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, target_image_pil):
        """

        :param target_image_pil: Target image as a PIL Image in RGB mode
        """
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(vgg.features.children())[:18])
        self.feature_extractor.eval()

        with torch.no_grad():
            self.target_features = self.feature_extractor(
                self._transform(target_image_pil).unsqueeze(0)
            )

    def get_fitness(self, individuals):
        """Assign fitness scores based on VGG feature map similarity against the target.

        :param individuals: List of Individual objects to evaluate
        """
        dirty = [ind for ind in individuals if ind.dirty]
        if not dirty:
            return

        batch = torch.stack([self._transform(ind.genotype) for ind in dirty])
        with torch.no_grad():
            features = self.feature_extractor(batch)

        target = self.target_features.expand(len(dirty), -1, -1, -1)
        mse_values = torch.mean((target - features) ** 2, dim=[1, 2, 3])

        for ind, mse in zip(dirty, mse_values):
            ind.set_fitness(1 / (1 + mse.item()))


class MobileNetPerceptualFitness():
    """Fitness evaluator using perceptual feature maps from a pretrained MobileNetV3 network."""

    DOWNSCALED_SIZE = (224, 224)

    _transform = transforms.Compose([
        transforms.Resize(DOWNSCALED_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, target_image_pil):
        """

        :param target_image_pil: Target image as a PIL Image in RGB mode
        """
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        self.feature_extractor.eval()

        with torch.no_grad():
            self.target_features = self.feature_extractor(
                self._transform(target_image_pil).unsqueeze(0)
            )

    def get_fitness(self, individuals):
        """Assign fitness scores based on MobileNet feature map similarity against the target.

        :param individuals: List of Individual objects to evaluate
        """
        dirty = [ind for ind in individuals if ind.dirty]
        if not dirty:
            return

        batch = torch.stack([self._transform(ind.genotype) for ind in dirty])
        with torch.no_grad():
            features = self.feature_extractor(batch)

        target = self.target_features.expand(len(dirty), -1, -1, -1)
        mse_values = torch.mean((target - features) ** 2, dim=[1, 2, 3])

        for ind, mse in zip(dirty, mse_values):
            ind.set_fitness(1 / (1 + mse.item()))


FITNESS = {
    "RGBMSE": RGBMSEFitness,
    "SSIM": SSIMFitness,
    "LABMSEFitness": LABMSEFitness,
    "LABSSIMFitness": LABSSIMFitness,
    "CIEDE2000Fitness": CIEDE2000Fitness,
    "VGGPerceptualFitness": VGGPerceptualFitness,
    "MobileNetPerceptualFitness": MobileNetPerceptualFitness,
}
