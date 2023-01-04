import argparse
import os
from .src.common import Sentinel2A

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--directory",
        required=True,
        help="Sentinel2A data directory to be processed"
    )

    parser.add_argument(
        "--rgb_image",
        required=False,
        help="Output sRGB image to file",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        required=False,
        help="Extract n pixel values from the data",
        type=int
    )

    parser.add_argument(
        "-r",
        "--resolution",
        required=False,
        help="Ground resolution to use ['10m', '20m', '60m']",
        default="10m",
        type=str
    )

    args = parser.parse_args()

    sentinel_data = Sentinel2A(os.path.abspath(args.directory))

    if args.rgb_image:
        image = sentinel_data.create_srgb(resolution=args.resolution)
        image.save(args.rgb_image)

    if args.n_samples:
        samples = sentinel_data.get_random_samples(args.n_samples)
        print(samples)
