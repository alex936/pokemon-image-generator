import os

from PIL import Image

from settings import POKEMON_DIR


def get_pokemons(size):
    pokemons = []
    file_paths = os.listdir(POKEMON_DIR)

    for path in file_paths:
        pokemon = Image.open(os.path.join(POKEMON_DIR, path)).resize((size, size), Image.LANCZOS)
        pokemons.append(pokemon)
    return pokemons