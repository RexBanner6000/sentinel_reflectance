import os
import re
import ssl
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pycountry_convert as pc
import rasterio
from geopy.geocoders import Nominatim
from PIL import Image
from skimage.color import xyz2rgb
from skimage.transform import resize

from .constants import CIE_M


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images: Dict = None
        self.mask: np.ndarray = None
        self.solar_irradiance: Dict = None
        self.band_mapping: Dict = None
        self.bounding_coords: Dict = None
        self.country: str = None
        self.country_code: str = None
        self.capture_date: str = None
        self.continent: str = None
        self.boa_offset: Dict = None
        self.boa_quantification: int = 10_000

        self.populate_metadata()

    def populate_metadata(self):
        self.images = self.get_image_files()
        self.band_mapping = self.get_physical_band_mapping()
        self.boa_offset = self.get_boa_offset()
        quants = self.get_quantification_values()
        self.boa_quantification = quants["BOA"]
        self.solar_irradiance = self.get_solar_irradiance()

        self.bounding_coords = self.get_bounding_coords()
        self.reverse_geocode_location()

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
            r"(?:<gmd:(\w*)Bound(\w*)>\n\s*<gco:Decimal>(\d{1,3}\.\d*)</gco:Decimal>)"
        )
        raw_str = self.read_text_file("INSPIRE.xml")
        bounding_coords = {}
        groups = re.findall(coords_regex, raw_str)

        for group in groups:
            if not bounding_coords.get(group[0]):
                bounding_coords[group[0]] = float(group[2])
        return bounding_coords

    def reverse_geocode_location(self):
        top_left = (self.bounding_coords["north"], self.bounding_coords["east"])
        bottom_right = (self.bounding_coords["south"], self.bounding_coords["west"])

        centre_coords = tuple(map(np.mean, zip(*(top_left, bottom_right))))
        #TODO: Fix SSL request so this isnt needed
        ssl._create_default_https_context = ssl._create_unverified_context

        locator = Nominatim(user_agent="sent2ref", scheme="https")
        location = locator.reverse(centre_coords)
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

    def create_mask(self, bands: Tuple[int] = (4, 5)):
        scl = rasterio.open(os.path.join(self.path, self.images["SCL"]["20m"])).read(1)

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
            linear_rgb[:, :, c] = (bands[band].read(1) + self.boa_offset[band])
            c += 1
        linear_rgb /= 4000
        return linear_rgb

    def create_band_image(self, bandId: str = "B08", resolution: str = "20m"):
        band = rasterio.open(
            os.path.join(self.path, self.images[bandId][resolution])
        )

        band_data = band.read(1) + self.boa_offset[bandId]
        return band_data / 4000

    def create_srgb(self, resolution: str = "10m"):
        rgb = self.create_linear_rgb(resolution=resolution)
        xyz = np.dot(rgb.reshape((-1, 3)), CIE_M).reshape(rgb.shape)
        srgb = xyz2rgb(xyz) * 255
        return Image.fromarray(srgb.astype("uint8"))

    def get_random_samples(self, n: int = 10_000, mask_bands: Tuple[int] = (4, 5)):
        rgb = self.create_linear_rgb(resolution="10m")
        mask = self.create_mask(mask_bands)
        mask = resize(mask, (rgb.shape[0], rgb.shape[1])).reshape(-1)
        rgb = rgb.reshape((-1, 3))
        rgb_samples = rgb[mask == 1, :]
        np.random.shuffle(rgb_samples)
        if mask.sum() < n:
            return rgb_samples
        else:
            return rgb_samples[:n, :]


if __name__ == "__main__":
    sent = Sentinel2A("D:\datasets\sentinel2a\S2B_MSIL2A_20220824T083559_N0400_R064_T36UYA_20220824T100829.SAFE")
    print(sent.get_random_samples(10))
    sent.create_srgb()
