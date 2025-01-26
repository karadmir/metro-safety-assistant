from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime
from PIL import Image


class UNet:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, image: np.array, conf: float = 0.5):
        img = cv2.resize(image, (224, 224))
        img = img.astype('float32')
        img = img / 127.5 - 1
        img = img[np.newaxis, ...]
        input_name = self.session.get_inputs()[0].name
        input_data = img
        out = self.session.run(None, {input_name: np.array(input_data)})[0][0]
        out = np.where(out > conf, 255, 0)
        out = out.astype('uint8')
        out = cv2.resize(out, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return out


yolo_pose = YOLO('models/yolo11s-pose.pt', task='pose')
yolo_seg = YOLO('models/yolo11s-seg.pt', task='segment')
unet = UNet('models/unet.onnx')


def detect_objects(frame):
    result = unet.predict(frame, conf=0.25)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # flat
    contours = np.concatenate(contours)
    hull = cv2.convexHull(contours)
    # fit line
    vx, vy, x, y = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
    # get as ax+b
    a = vy / vx
    b = y - a * x

    # draw
    lefty = int((-x * vy / vx) + y)
    righty = int(((frame.shape[1] - x) * vy / vx) + y)
    safety_line = np.array([[frame.shape[1] - 1, righty], [0, lefty]])

    safety_line = hull
    safety_line = np.squeeze(safety_line, axis=1)

    objs = []

    objs.append(('line', safety_line))

    result = yolo_pose.predict(frame, imgsz=320, verbose=False)[0]
    for i in range(len(result.boxes)):
        bbox = result.boxes[i].xyxy.numpy()[0]
        kps = result.keypoints[i].xy.numpy()[0]
        kps = np.concatenate(kps)
        objs.append(('person', np.concatenate([bbox, kps])))

    result = yolo_seg.predict(frame, imgsz=320, verbose=False, conf=0.6)[0]
    for i in range(len(result.boxes)):
        cls = int(result.boxes[i].cls)
        cls = result.names[cls]
        if cls == 'person':
            continue
        objs.append((cls, result.boxes[i].xyxy.numpy()))

    return objs


if __name__ == '__main__':
    test_image = Image.open('test.jpg')
    # resize in half
    test_image = test_image.resize((test_image.width // 2, test_image.height // 2))
    test_image = np.array(test_image)
    output = detect_objects(test_image)