<H1 align="center">
RF-DETR Object Detection with DeepSORT Tracking </H1>

## Steps to run Code

- Clone the repository
```
git clone https://github.com/farukalamai/rfdetr-deepsort-object-tracking.git
```
```
cd rfdetr-deepsort-object-tracking
```
- Install the dependecies
```
pip install -r requirements.txt

```
- Run the code with mentioned command below.

- For yolov8 object detection + Tracking
```
python predict.py model=yolov8l.pt source="test3.mp4" show=True
```
