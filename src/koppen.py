import numpy as np
from pathlib import Path


def read_climate_data(filename: Path):
    koppen = np.genfromtxt(
        filename,
        dtype=[("latitude", "f8"), ("longitude", "f8"), ("p1901-2010", "U3")], names=True
    )
    return koppen


def create_climate_labels(climates):
    climate_labels = {}
    for i, climate in enumerate(np.unique(climates)):
        climate_labels[climate] = i
    return climate_labels


def get_coord_climate(
        latitude: float, longitude: float, koppen
):
    climate_labels = create_climate_labels(koppen["p1901_2010"])

    z = np.vectorize(climate_labels.get)(koppen["p1901_2010"])

    coord_climate_id = lookup_nearest(
        latitude, longitude, koppen["latitude"], koppen["longitude"], z
    )

    key = next(
        key for key, value in climate_labels.items() if value == coord_climate_id
    )

    return key


def lookup_nearest(x0, y0, x, y, z):
    xi = np.abs(x-x0)
    yi = np.abs(y-y0)
    zi = (xi ** 2 + yi ** 2).argmin()
    return z[zi]
