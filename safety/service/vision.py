from ultralytics import YOLO
import cv2
import numpy as np
import onnxruntime
from PIL import Image


class UNet:
    def __init__(self, model_path: str):
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


def detect_objects(frame: np.array) -> list:
    result = unet.predict(frame, conf=0.25)
    # save result
    cv2.imwrite('segmentation.jpg', result)
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # flat
    contours = np.concatenate(contours)
    hull = cv2.convexHull(contours)

    # draw hull
    #cv2.polylines(frame, [hull], isClosed=True, color=(255, 0, 0), thickness=2)

    # fit line
    vx, vy, x, y = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
    # given this line, separate points orthogonally
    left_pts = []
    right_pts = []
    for pt in hull:
        pt = pt[0]
        if (pt[1] - y) / vy > (pt[0] - x) / vx:
            left_pts.append(pt)
        else:
            right_pts.append(pt)

    # fit lines
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)

    left_vx, left_vy, left_x, left_y = cv2.fitLine(left_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    right_vx, right_vy, right_x, right_y = cv2.fitLine(right_pts, cv2.DIST_L2, 0, 0.01, 0.01)

    # get as ax+b
    left_k = left_vy / left_vx
    left_b = left_y - left_k * left_x
    right_k = right_vy / right_vx
    right_b = right_y - right_k * right_x

    # draw lines on black image
    # img = np.zeros_like(frame)
    # cv2.line(img, (0, int(left_b)), (frame.shape[1], int(left_k * frame.shape[1] + left_b)), (255, 255, 255), 2)
    # cv2.line(img, (0, int(right_b)), (frame.shape[1], int(right_k * frame.shape[1] + right_b)), (255, 255, 255), 2)
    # # save image
    # cv2.imwrite('lines.jpg', img)

    objs = []

    objs.append(('line', (left_k, left_b)))
    objs.append(('line', (right_k, right_b)))

    result = yolo_pose.predict(frame, imgsz=320, verbose=False)[0]
    for i in range(len(result.boxes)):
        bbox = result.boxes[i].xyxy.numpy()[0]
        kps = result.keypoints[i].xy.numpy()[0]
        kps = np.concatenate(kps)
        objs.append(('person', np.concatenate([bbox, kps])))

    result = yolo_seg.predict(frame, imgsz=320, verbose=False, conf=0.65)[0]
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