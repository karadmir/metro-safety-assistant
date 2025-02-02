'''
This module contains the logic for assessing the safety of pedestrians in a subway station based on yellow line and train proximity
'''

import numpy as np

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Person:
    def __init__(self, bbox: np.array, keypoints: np.array, safe: bool):
        self.bbox = bbox
        self.keypoints = keypoints
        self.safe = safe
        self.leftfoot = Point(self.keypoints[30], self.keypoints[31])
        self.rightfoot = Point(self.keypoints[32], self.keypoints[33])


class SafetyLine:
    def __init__(self, k: float, b: float):
        self.k = k
        self.b = b


class Train:
    def __init__(self, bbox: np.array):
        self.bbox = bbox


class Station:
    def __init__(self, safety_lines: list[SafetyLine], train: [Train, None], people: list[Person]):
        self.safety_lines = safety_lines
        self.train = train
        self.people = people


def assess_safety(station: Station):
    if abs(station.safety_lines[0].k) < abs(station.safety_lines[1].k):
        upperlimit = station.safety_lines[0]
        lowerlimit = station.safety_lines[1]
    else:
        upperlimit = station.safety_lines[1]
        lowerlimit = station.safety_lines[0]

    for person in station.people:
        if station.train:
            person.safe = True
        else:
            if (person.leftfoot.y < (upperlimit.k * person.leftfoot.x + upperlimit.b) and person.leftfoot.y > (
                    lowerlimit.k * person.leftfoot.x + lowerlimit.b)) or (
                    person.rightfoot.y < (upperlimit.k * person.rightfoot.x + upperlimit.b) and person.rightfoot.y > (
                    lowerlimit.k * person.rightfoot.x + lowerlimit.b)):
                person.safe = False
            else:
                person.safe = True


if __name__ == '__main__':
    import cv2
    import os
    import json

    image_dir = 'test'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        test_image = cv2.imread(os.path.join(image_dir, image_file))

        frame = np.array(test_image)
        draw = frame

        w = frame.shape[1]
        h = frame.shape[0]

        with open('test/' + base_name + '.json') as f:
            data = json.load(f)

        safety_lines = [SafetyLine(line['k'], line['b']) for line in data['safety_lines']]
        if data['train']:
            train = Train(np.array(data['train']['bbox']))
        else:
            train = None
        people = [Person(np.array(person['bbox']), np.array(person['keypoints']), person['safe']) for person in data['people']]

        station = Station(safety_lines, train, people)

        assess_safety(station)

        for line in station.safety_lines:
            a = line.k
            b = line.b
            x1 = 0
            y1 = int(b)
            x2 = w
            y2 = int(a * x2 + b)
            if y2 > h:
                y2 = h
                x2 = int((y2 - b) / a)
            elif y2 < 0:
                y2 = 0
                x2 = int((y2 - b) / a)
            if y1 > h:
                y1 = h
                x1 = int((y1 - b) / a)
            elif y1 < 0:
                y1 = 0
                x1 = int((y1 - b) / a)
            cv2.line(draw, (x1, y1), (x2, y2), (0, 255, 255), 2)

        for person in people:
            if person.safe:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            tr = (int(person.bbox[0]), int(person.bbox[1]))
            br = (int(person.bbox[2]), int(person.bbox[3]))
            cv2.rectangle(draw, tr, br, color, 2)
            cv2.circle(draw, (int(person.leftfoot.x), int(person.leftfoot.y)), 2, (255, 255, 255), 2)
            cv2.circle(draw, (int(person.rightfoot.x), int(person.rightfoot.y)), 2, (255, 255, 255), 2)

        cv2.imshow('frame', draw)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
