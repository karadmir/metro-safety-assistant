'''
This script is used to generate masks from the annotations file. The annotations file is in the format of json (COCO format)
Author: Radmir Kadyrov
'''

import json
import argparse
import os

import cv2
import numpy as np


argparser = argparse.ArgumentParser(description='Generate masks from annotation file')
argparser.add_argument('annotations_file', help='Path to the annotations file')
argparser.add_argument('output_dir', help='Output directory for masks')
args = argparser.parse_args()

with open(args.annotations_file) as f:
    annotations = json.load(f)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

for image in annotations['images']:
    image_path = image['file_name']
    image_id = image['id']
    image_width = image['width']
    image_height = image['height']

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for annotation in annotations['annotations']:
        if annotation['category_id'] != 1:
            continue
        if annotation['image_id'] == image_id:
            segmentation = annotation['segmentation']
            segmentation = np.array(segmentation, dtype=np.int32)
            segmentation = segmentation.reshape(-1, 1, 2)
            cv2.fillPoly(mask, [segmentation], 255)

    mask_path = os.path.join(args.output_dir, os.path.basename(image_path))
    cv2.imwrite(mask_path, mask)
    print(f'Saved mask for {image_path} to {mask_path}')





