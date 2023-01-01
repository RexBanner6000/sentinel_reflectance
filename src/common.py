import os
import rasterio
import re

import numpy as np
from pathlib import Path
from typing import Set


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images = self.get_image_files()
        self.mask: np.ndarray = None

    def get_image_files(self):
        img_regex = re.compile(
            r"(?:<IMAGE_FILE>(GRANULE/.*_(.{1,3})_(\d{1,2}m))</IMAGE_FILE>)"
        )
        with open(os.path.join(self.path, "MTD_MSIL2A.xml"), "r") as fp:
            raw_str = fp.read()

        images = {}
        groups = re.findall(img_regex, raw_str)
        for group in groups:
            if not images.get(group[1]):
                images[group[1]] = {}
            if not images[group[1]].get(group[2]):
                images[group[1]][group[2]] = group[0] + ".jp2"

        return images

    def create_mask(self, bands: Set[int] = (4, 5)):
        scl = rasterio.open(
            os.path.join(self.path, self.images["SCL"]["20m"])
        ).read(1)

        mask = np.zeros(scl.shape)

        for band in bands:
            mask[scl == band] = 1

        return mask

    def create_linear_rgb(self, resolution: str = "10m"):
        bands = {}
        for band in ["B04", "B03", "B02"]:
            bands[band] = rasterio.open(
                os.path.join(self.path, self.images["B04"][resolution])
            )

        linear_rgb = np.zeros((bands["B04"].height, bands["B04"].width, 3))

        for i, band in enumerate(bands.values()):
            linear_rgb[..., i] = band.read(1)

        linear_rgb /= 4000

        return linear_rgb


