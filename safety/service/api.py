import os

from fastapi import FastAPI, File, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2

from safety.service.analysis import SafetyLine, Train, Person
from vision import detect_objects
from analysis import assess_safety, Station

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/process')
def process(video: bytes = File(...)):
    with open('video.mp4', 'wb') as f:
        f.write(video)

    cap = cv2.VideoCapture('video.mp4')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert to RGB
        draw = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        objects = detect_objects(frame)

        safety_lines = []
        train = None
        people = []

        for obj in objects:
            if obj[0] == 'line':
                safety_lines.append(SafetyLine(obj[1][0], obj[1][1]))
            elif obj[0] == 'train':
                train = Train(obj[1])
            elif obj[0] == 'person':
                p = Person(obj[1][:4], obj[1][4:], False)
                people.append(p)

        station = Station(safety_lines, train, people)

        assess_safety(station)

        for person in people:
            if person.safe:
                color = (255, 255, 255)
            else:
                color = (0, 0, 255)
            tr = (int(person.bbox[0]), int(person.bbox[1]))
            br = (int(person.bbox[2]), int(person.bbox[3]))
            cv2.rectangle(draw, tr, br, color, 2)

        out.write(draw)

    cap.release()
    out.release()

    os.system("ffmpeg -y -i output.mp4 -vcodec libx264 -f mp4 result.mp4")

    return 'done'


@app.get('/result/{result_id}/video')
def result(result_id: int) -> Response:
    with open('result.mp4', 'rb') as f:
        return Response(content=f.read(), media_type='video/mp4')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, ssl_keyfile='key.pem', ssl_certfile='cert.pem')