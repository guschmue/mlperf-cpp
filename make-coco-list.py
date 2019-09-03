
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# pylint: disable=missing-docstring

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="coco directory")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    image_list = os.path.join(args.input, "annotations/instances_val2017.json")

    with open(image_list, "r") as f:
        coco = json.load(f)
    for i in coco["images"]:
        print("NHWC/val2017/" + i["file_name"])

if __name__ == "__main__":
    main()
