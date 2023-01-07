import os
import re
import ssl
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
import psycopg2
import pycountry_convert as pc
import rasterio
from geopy.geocoders import Nominatim
from PIL import Image
from skimage.color import xyz2rgb

from src.constants import CIE_M
from src.koppen_climate import get_coord_climate, read_climate_data


class Season(Enum):
    SPRING = "SPRING"
    SUMMER = "SUMMER"
    AUTUMN = "AUTUMN"
    WINTER = "WINTER"
    UNKNOWN = "UNKNOWN"


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images: Dict = None
        self.mask: np.ndarray = None
        self.solar_irradiance: Dict = None
        self.band_mapping: Dict = None
        self.bounding_coords: Dict = None
        self.centre_coords: tuple = (np.nan, np.nan)
        self.country: str = None
        self.country_code: str = None
        self.capture_date: str = None
        self.continent: str = None
        self.boa_offset: Dict = None
        self.boa_quantification: int = 10_000
        self.processing_level: str = "NA"
        self.product_type: str = "NA"
        self.product_uri: str = "NA"
        self.season: Season = Season.UNKNOWN

        self.populate_metadata()

    def populate_metadata(self):
        self.images = self.get_image_files()
        self.band_mapping = self.get_physical_band_mapping()
        self.boa_offset = self.get_boa_offset()
        quants = self.get_quantification_values()
        self.boa_quantification = quants["BOA"]
        self.solar_irradiance = self.get_solar_irradiance()

        self.bounding_coords = self.get_bounding_coords()
        self.centre_coords = get_centre_coords(self.bounding_coords)
        self.reverse_geocode_location()

        self.capture_date = self.get_product_info("PRODUCT_START_TIME")
        self.product_type = self.get_product_info("PRODUCT_TYPE")
        self.processing_level = self.get_product_info("PROCESSING_LEVEL")
        self.product_uri = self.get_product_info("PRODUCT_URI")
        self.season = get_season(self.capture_date.split("T")[0], self.centre_coords[0])

    def get_product_info(self, param: str):
        capture_regex = re.compile(rf"(?:<{param}>(.*)</\w*>)")
        raw_str = self.read_text_file("MTD_MSIL2A.xml")
        match = re.search(capture_regex, raw_str)
        return match.group(1)

    def get_boa_offset(self):
        boa_offset_regex = re.compile(
            r"(?:<BOA_ADD_OFFSET band_id=\"(\d{1,2})\">(-?\d{1,4})</BOA_ADD_OFFSET>)"
        )
        raw_str = self.read_text_file("MTD_MSIL2A.xml")
        boa_offsets = {}
        groups = re.findall(boa_offset_regex, raw_str)

        for group in groups:
            boa_offsets[self.band_mapping[group[0]]] = int(group[1])

        return boa_offsets

    def get_quantification_values(self):
        quantification_regex = re.compile(
            r"(?:<(\w{1,3})_QUANTIFICATION_VALUE unit=\"\S*\">(\d+(?:\.\d+)?)</\w{1,3}_QUANTIFICATION_VALUE>)"
        )
        raw_str = self.read_text_file("MTD_MSIL2A.xml")
        quantification_values = {}
        groups = re.findall(quantification_regex, raw_str)
        for group in groups:
            quantification_values[group[0]] = float(group[1])
        return quantification_values

    def get_bounding_coords(self):
        coords_regex = re.compile(
            r"(?:<gmd:(\w*)Bound(\w*)>\n\s*<gco:Decimal>(-?\d{1,3}\.\d*)</gco:Decimal>)"
        )
        raw_str = self.read_text_file("INSPIRE.xml")
        bounding_coords = {}
        groups = re.findall(coords_regex, raw_str)

        for group in groups:
            if not bounding_coords.get(group[0]):
                bounding_coords[group[0]] = float(group[2])
        return bounding_coords

    def reverse_geocode_location(self):
        # TODO: Fix SSL request so this isnt needed
        ssl._create_default_https_context = ssl._create_unverified_context

        locator = Nominatim(user_agent="sent2ref", scheme="https")
        location = locator.reverse(self.centre_coords)
        self.country = location.raw["address"]["country"]
        self.country_code = location.raw["address"]["country_code"].upper()
        self.continent = pc.country_alpha2_to_continent_code(self.country_code)

    def get_physical_band_mapping(self):
        physical_band_regex = re.compile(
            r"(?:<Spectral_Information bandId=\"(\d{1,2})\" physicalBand=\"(\w)(\d+(?:\w)?)\">)"
        )

        raw_str = self.read_text_file("MTD_MSIL2A.xml")
        band_mapping = {}
        groups = re.findall(physical_band_regex, raw_str)

        for group in groups:
            try:
                band_mapping[group[0]] = group[1] + f"{int(group[2]):02d}"
            except ValueError:
                band_mapping[group[0]] = group[1] + f"{(group[2])}"

        return band_mapping

    def read_text_file(self, filename: str):
        with open(os.path.join(self.path, filename), "r") as fp:
            raw_str = fp.read()
        return raw_str

    def get_solar_irradiance(self):
        irradiance_regex = re.compile(
            r"(?:<SOLAR_IRRADIANCE bandId=\"(\d{1,2})\") unit=\"(.*)\">(\d*\.\d{1,2})</SOLAR_IRRADIANCE>"
        )
        raw_str = self.read_text_file("MTD_MSIL2A.xml")
        solar_irr = {}
        groups = re.findall(irradiance_regex, raw_str)

        for group in groups:
            if not solar_irr.get(group[0]):
                solar_irr[self.band_mapping[group[0]]] = float(group[2])

        return solar_irr

    def get_image_files(self):
        img_regex = re.compile(
            r"(?:<IMAGE_FILE>(GRANULE/.*_(.{1,3})_(\d{1,2}m))</IMAGE_FILE>)"
        )
        raw_str = self.read_text_file("MTD_MSIL2A.xml")

        images = {}
        groups = re.findall(img_regex, raw_str)
        for group in groups:
            if not images.get(group[1]):
                images[group[1]] = {}
            if not images[group[1]].get(group[2]):
                images[group[1]][group[2]] = group[0] + ".jp2"

        return images

    def create_mask(self, scl: np.ndarray, bands: Tuple[int] = (2, 4, 5, 11)):
        mask = np.zeros(scl.shape)
        for band in bands:
            mask[scl == band] = 1
        return mask

    def create_linear_rgb(self, resolution: str = "10m"):
        bands = {}
        for band in ["B02", "B03", "B04"]:
            bands[band] = rasterio.open(
                os.path.join(self.path, self.images[band][resolution])
            )

        linear_rgb = np.zeros((bands["B04"].height, bands["B04"].width, 3))
        c = 0
        for band in ["B02", "B03", "B04"]:
            linear_rgb[:, :, c] = bands[band].read(1) + self.boa_offset[band]
            c += 1
        linear_rgb /= self.boa_quantification + self.boa_offset[band]
        return linear_rgb

    def create_band_image(self, bandid: str = "B08", resolution: str = "10m"):
        band = rasterio.open(os.path.join(self.path, self.images[bandid][resolution]))

        band_data = band.read(1) + self.boa_offset[bandid]
        return band_data / (self.boa_quantification + self.boa_offset[bandid])

    def create_srgb(self, resolution: str = "10m"):
        rgb = self.create_linear_rgb(resolution=resolution)
        xyz = np.dot(rgb.reshape((-1, 3)), CIE_M).reshape(rgb.shape)
        srgb = xyz2rgb(xyz) * 255
        return Image.fromarray(srgb.astype("uint8"))

    def get_random_samples(
        self, n: int = 10_000, mask_bands: Tuple[int] = (2, 4, 5, 11)
    ):
        rgb = self.create_linear_rgb(resolution="10m")
        nir = self.create_band_image(bandid="B08", resolution="10m").reshape((-1, 1))
        scl = rasterio.open(os.path.join(self.path, self.images["SCL"]["20m"])).read(1)
        scl = np.repeat(np.repeat(scl, 2, axis=0), 2, axis=1)
        mask = self.create_mask(scl, mask_bands).reshape(-1)
        scl = scl.reshape((-1, 1))
        rgb = rgb.reshape((-1, 3))

        data = np.append(rgb, nir, axis=1)
        data = np.append(data, scl, axis=1)
        samples = data[mask == 1, :]
        idx = np.random.randint(0, len(samples), n)
        del data, rgb, nir, scl, mask

        if len(samples) < n:
            return samples
        else:
            return samples[idx, :]

    def samples_to_db(self, n: int = 10_000):
        print(f"Getting {n} samples...")
        samples = self.get_random_samples(n)

        print("Getting climate data...")
        koppen = read_climate_data("./koppen_1901-2010.tsv")
        climate = get_coord_climate(
            self.centre_coords[0], self.centre_coords[1], koppen
        )

        print("Connecting to DB...")
        connection = self._connect_to_db()
        cur = connection.cursor()

        print("Generating SQL...")
        for sample in samples:
            sql = self._generate_sql(sample, climate)
            cur.execute(sql)
        print("Committing SQL to DB...")
        connection.commit()
        print("Done!\n")

    def _connect_to_db(self):
        return psycopg2.connect(
            "dbname='postgres' user='postgres' host='localhost' password='postgres'"
        )

    def _generate_sql(self, sample: np.array, climate: str, table: str = "sentinel2a"):
        sql = (
            f"INSERT INTO {table} "
            f"(uuid,product_uri,country,continent,capture,b02,b03,b04,b08,season,climate,classification) "
            f"VALUES ('{uuid4().hex}', '{self.product_uri}', '{self.country_code}', "
            f"'{self.continent}', '{self.capture_date}', '{sample[0]}', '{sample[1]}', "
            f"'{sample[2]}', '{sample[3]}', '{self.season.value}', '{climate}', '{int(sample[4])}')"
        )

        return sql


def get_season(date: str, latitude: float):
    spring = range(60, 151)
    summer = range(152, 243)
    autumn = range(244, 334)

    date = datetime.strptime(date, "%Y-%m-%d")
    if latitude < 0:
        date += timedelta(days=180)

    day_of_year = date.timetuple().tm_yday

    if day_of_year in spring:
        season = Season.SPRING
    elif day_of_year in summer:
        season = Season.SUMMER
    elif day_of_year in autumn:
        season = Season.AUTUMN
    else:
        season = Season.WINTER

    return season


def get_centre_coords(coords_dict: dict):
    top_left = (coords_dict["north"], coords_dict["east"])
    bottom_right = (coords_dict["south"], coords_dict["west"])
    coords = list(map(np.mean, zip(*(top_left, bottom_right))))
    if coords[1] > 180:
        coords[1] -= 360
    return tuple(coords)


if __name__ == "__main__":
    for img in os.listdir(r"D:\datasets\sentinel2a\\"):
        print(f"Reading {img}...")
        sentinel = Sentinel2A(rf"D:\datasets\sentinel2a\{img}")
        sentinel.samples_to_db(100_000)
        del sentinel
