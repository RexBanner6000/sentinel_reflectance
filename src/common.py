import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from PIL import Image
from skimage.color import xyz2rgb

from .constants import CIE_M


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images: Dict = None
        self.mask: np.ndarray = None
        self.solar_irradiance: Dict = None
        self.band_mapping: Dict = None

        self.populate_metadata()

    def populate_metadata(self):
        self.images = self.get_image_files()
        self.band_mapping = self.get_physical_band_mapping()
        self.solar_irradiance = self.get_solar_irradiance()

    def get_physical_band_mapping(self):
        physical_band_regex = re.compile(
            r"(?:<Spectral_Information bandId=\"(\d{1,2})\" physicalBand=\"(.{1,3})\">)"
        )

        raw_str = self.read_xml_file()
        band_mapping = {}
        groups = re.findall(physical_band_regex, raw_str)

        for group in groups:
            if not band_mapping.get(group[0]):
                band_mapping[group[0]] = group[1]

        return band_mapping

    def read_xml_file(self):
        with open(os.path.join(self.path, "MTD_MSIL2A.xml"), "r") as fp:
            raw_str = fp.read()
        return raw_str

    def get_solar_irradiance(self):
        irradiance_regex = re.compile(
            r"(?:<SOLAR_IRRADIANCE bandId=\"(\d{1,2})\") unit=\"(.*)\">(\d*\.\d{1,2})</SOLAR_IRRADIANCE>"
        )
        raw_str = self.read_xml_file()
        solar_irr = {}
        groups = re.findall(irradiance_regex, raw_str)

        for group in groups:
            if not solar_irr.get(group[0]):
                solar_irr[group[0]] = float(group[2])

        return solar_irr

    def get_image_files(self):
        img_regex = re.compile(
            r"(?:<IMAGE_FILE>(GRANULE/.*_(.{1,3})_(\d{1,2}m))</IMAGE_FILE>)"
        )
        raw_str = self.read_xml_file()

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

    def create_linear_rgb(self, resolution: str = "20m"):
        bands = {}
        for band in ["B02", "B03", "B04"]:
            bands[band] = rasterio.open(
                os.path.join(self.path, self.images[band][resolution])
            )

        linear_rgb = np.zeros((bands["B04"].height, bands["B04"].width, 3))
        for i, band in enumerate(bands.values()):
            linear_rgb[:, :, i] = band.read(1)
        linear_rgb /= 4000
        return linear_rgb

    def create_srgb(self):
        rgb = self.create_linear_rgb()
        xyz = np.dot(rgb.reshape((-1, 3)), CIE_M).reshape(rgb.shape)
        srgb = xyz2rgb(xyz) * 255
        return Image.fromarray(srgb.astype("uint8"))

    def get_random_samples(self, n: int = 10_000, mask_bands: Tuple[int] = (4, 5)):
        rgb = self.create_linear_rgb().reshape((-1, 3))
        mask = self.create_mask(mask_bands).reshape(-1).astype("bool")
        rgb_samples = rgb[mask, :]
        np.random.shuffle(rgb_samples)
        if mask.sum() < n:
            return rgb_samples
        else:
            return rgb_samples[:n, :]
