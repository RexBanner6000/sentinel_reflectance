from src.common import get_season, Season


def test_get_season():
    assert get_season("2022-01-01", 50) == Season.WINTER
