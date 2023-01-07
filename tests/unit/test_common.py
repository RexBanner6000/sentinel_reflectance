from src.common import Season, get_centre_coords, get_season


def test_get_season():
    assert get_season("2022-01-01", 50) == Season.WINTER
    assert get_season("2022-01-01", -50) == Season.SUMMER
    assert get_season("2022-04-01", 50) == Season.SPRING
    assert get_season("2022-07-01", 50) == Season.SUMMER
    assert get_season("2022-10-01", 50) == Season.AUTUMN


def test_get_centre_coords():
    coords_dict = {
        "north": 50,
        "south": 49,
        "east": 2,
        "west": 1
    }

    assert get_centre_coords(coords_dict) == (49.5, 1.5)
