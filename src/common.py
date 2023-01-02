import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from PIL import Image
from skimage.color import xyz2rgb

from src.constants import CIE_M


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images: Dict = None
        self.mask: np.ndarray = None

    def populate_metadata(self):
        self.images = self.get_image_files()

    def read_xml_file(self):
        with open(os.path.join(self.path, "MTD_MSIL2A.xml"), "r") as fp:
            raw_str = fp.read()
        return raw_str

    def get_solar_irradiance(self):
        irradiance_regex = re.compile(
            r"(?:<SOLAR_IRRADIANCE bandId=\"(\d{1,2})\") unit=\"(.*)\">\d*\.\d{1,2}</SOLAR_IRRADIANCE>"
        )
        raw_str = self.read_xml_file()
        solar_irr = {}
        groups = re.findall(irradiance_regex, raw_str)

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
        scl = rasterio.open(
            os.path.join(self.path, self.images["SCL"]["20m"])
        ).read(1)

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


if __name__ == "__main__":
    sent = Sentinel2A("./S2B_MSIL2A_20221205T085239_N0400_R107_T36UXA_20221205T102012.SAFE")
    sent.get_random_samples(100)
