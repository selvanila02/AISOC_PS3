import os, time, csv
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
from picamera2 import Picamera2
from datetime import datetime

MODEL_PATH = "/home/pi/best_320.onnx"
IMG_SIZE = 320
CONF_THRES = 0.40
IOU_THRES = 0.45
MAX_DET = 50

SAVE_DIR = "/home/pi/detections"
IMG_DIR = os.path.join(SAVE_DIR, "images")
CSV_PATH = os.path.join(SAVE_DIR, "log.csv")
os.makedirs(IMG_DIR, exist_ok=True)

app = Flask(__name__)

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cam = Picamera2()
cam.configure(cam.create_video_configuration(main={"size": (320, 320), "format": "RGB888"}))
cam.start()
time.sleep(1)

total_count = 0
last_save_time = 0

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "conf", "x1", "y1", "x2", "y2", "count"])

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter <= 0: 
        return 0.0
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def nms(dets, iou_th=0.45):
    if len(dets) == 0:
        return []
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best, d) < iou_th]
    return keep

def gen():
    global total_count, last_save_time
    t0 = time.time()
    frames = 0

    while True:
        frame = cam.capture_array()  # RGB 320x320
        img = frame.astype(np.float32) / 255.0
        x = img.transpose(2, 0, 1)[None, ...]

        out = session.run(None, {input_name: x})[0][0]  # (6300,6)

        dets = []
        for d in out:
            conf = float(d[4])
            if conf >= CONF_THRES:
                x1,y1,x2,y2 = map(float, d[:4])
                dets.append([x1,y1,x2,y2,conf,int(d[5])])

        dets = nms(dets, IOU_THRES)[:MAX_DET]

        for d in dets:
            x1,y1,x2,y2,conf,_ = d
            x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{conf:.2f}",(x1,max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        frames += 1
        fps = frames / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"FPS:{fps:.1f}  Det:{len(dets)}", (5,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # logging + save frame if detected (rate limit: 1 save per 1 sec)
        if len(dets) > 0 and (time.time() - last_save_time) > 1.0:
            best = max(dets, key=lambda x: x[4])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"{ts}.jpg"
            cv2.imwrite(os.path.join(IMG_DIR, img_name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            total_count += 1
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ts, best[4], int(best[0]), int(best[1]), int(best[2]), int(best[3]), total_count])

            print(f"Detected! Total Count: {total_count}")
            last_save_time = time.time()

        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/")
def home():
    return "<h2>RoadPi Live</h2><img src='/video' style='width:320px;max-width:100%;'/>"

@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting Flask on :8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)
