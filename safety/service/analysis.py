'''
This module contains the logic for assessing the safety of pedestrians in a subway station based on yellow line and train proximity
'''

import numpy as np


class Person:
    def __init__(self, bbox: np.array, keypoints: np.array):
        self.bbox = bbox
        self.keypoints = keypoints


class SafetyLine:
    def __init__(self, k: float, b: float):
        self.k = k
        self.b = b


class Train:
    def __init__(self, bbox: np.array):
        self.bbox = bbox


class Station:
    def __init__(self, safety_lines: list[SafetyLine], train: Train, people: list[Person]):
        self.safety_lines = safety_lines
        self.train = train
        self.people = people


def assess_safety(frame, station):
    print('Assessing safety...')
    pass


if __name__ == '__main__':
    assess_safety(None, None)