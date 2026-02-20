RoadPi package:
- stream_yolo.py (Flask + Picamera2 + ONNXRuntime)
- best_320.onnx (YOLOv5n nano exported)
- roadpi.service (systemd auto-start)

Run:
source /home/pi/yolovenv/bin/activate
python /home/pi/stream_yolo.py

Service:
sudo systemctl restart roadpi.service
sudo systemctl status roadpi.service --no-pager

Browser:
http://<PI_IP>:8000
