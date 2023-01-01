import os
import rasterio
import re

from pathlib import Path


class Sentinel2A:
    def __init__(self, path: Path):
        self.path = path
        self.images = self.get_image_files()

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
                images[group[1]][group[2]] = group[0]

        return images
