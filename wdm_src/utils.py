import numpy as np

ALTITUDES = np.arange(20, 251, 5)
DIRS = ['NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N']

def get_vector_cols(name: str) -> list:
    return [name + str(alt) for alt in ALTITUDES]
