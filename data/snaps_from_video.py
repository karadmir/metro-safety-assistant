'''
This script is used to extract frames from a video file with a given interval
Author: Radmir Kadyrov
'''

import os
import sys
import argparse
import random

import cv2


argparser = argparse.ArgumentParser(description='Extract frames from a video file.')
argparser.add_argument('video_file', help='Path to the video file.')
argparser.add_argument('output_dir', help='Path to the output directory.')
argparser.add_argument('interval', help='Interval to extract frames.', type=int)
args = argparser.parse_args()

video_file = args.video_file
output_dir = args.output_dir
interval = args.interval

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print('Error: Cannot open video file.')
    sys.exit(1)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_indexes = range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), interval)
for i in frame_indexes:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        print('Error: Cannot read frame.')
        break
    output_file = os.path.join(output_dir, 'frame_{:05d}.png'.format(random.randint(0, 99999)))
    cv2.imwrite(output_file, frame)