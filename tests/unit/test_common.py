from src.common import Season, get_season


def test_get_season():
    assert get_season("2022-01-01", 50) == Season.WINTER
    assert get_season("2022-01-01", -50) == Season.SUMMER
    assert get_season("2022-04-01", 50) == Season.SPRING
    assert get_season("2022-07-01", 50) == Season.SUMMER
    assert get_season("2022-10-01", 50) == Season.AUTUMN
