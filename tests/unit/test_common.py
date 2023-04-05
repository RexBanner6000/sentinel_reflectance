from unittest.mock import patch

import numpy as np

from src.common import Season, Sentinel2A, get_centre_coords, get_season


def test_get_season():
    assert get_season("2022-01-01", 50) == Season.WINTER
    assert get_season("2022-01-01", -50) == Season.SUMMER
    assert get_season("2022-04-01", 50) == Season.SPRING
    assert get_season("2022-07-01", 50) == Season.SUMMER
    assert get_season("2022-10-01", 50) == Season.AUTUMN


def test_get_centre_coords():
    coords_dict = {"north": 50, "south": 49, "east": 2, "west": 1}

    assert get_centre_coords(coords_dict) == (49.5, 1.5)


@patch("src.common.Sentinel2A.populate_metadata")
def test_latlon2utm(mock_populate_metadata):
    mock_populate_metadata.side_effect = [None]
    sentinel = Sentinel2A("./foo")

    eastings, northings = sentinel._convert_latlon2utm(50.6980, -2.2286)
    assert np.abs(eastings - 554479) < 50
    assert np.abs(northings - 5616526) < 50

    eastings, northings = sentinel._convert_latlon2utm(-31.1988, 136.825)
    assert np.abs(eastings - 673877) < 50
    assert np.abs(northings - 6546931) < 50

    eastings, northings = sentinel._convert_latlon2utm(49.9935, 36.2304)
    assert np.abs(eastings - 301497) < 50
    assert np.abs(northings - 5541584) < 50
