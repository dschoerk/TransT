
from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

from pathlib import Path
import cv2
import numpy as np

def init_tracker(img, bbox):

    # wget --no-check-certificate 'https://drive.google.com/uc?id=1Pq0sK-9jmbLAVtgB9-dPDc2pipCxYdM5' -O /transt.pth
    net_path = './transt.pth'  # path of the model

    # create model
    net = NetWithBackbone(net_path=net_path, use_gpu=False)
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)
    

    cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))
    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]
    init_info = {'init_bbox': gt_bbox_}
    tracker.initialize(img, init_info)

    return tracker

def track(tracker, img):
    outputs = tracker.track(img)
    prediction_bbox = outputs['target_bbox']

    left = prediction_bbox[0]
    top = prediction_bbox[1]
    right = prediction_bbox[0] + prediction_bbox[2]
    bottom = prediction_bbox[1] + prediction_bbox[3]
    return tracker, (top, left, bottom, right)


video_path = Path('./testimages')
frames = list(video_path.iterdir())


frames = [cv2.resize(cv2.imread(str(x)), (640, 480)) for x in frames]

r = cv2.selectROI(frames[0])
#r = (1053, 397, 28, 78)
print(r)



tracker = init_tracker(frames[0], (r[0], r[1], r[2], r[3]))

#print(type(tracker.net.net))

#tracker, bbox = track(tracker, frames[1])

#exit()

for i in range(1, len(frames)):
    tracker, bbox = track(tracker, frames[i])

    bbox = [int(x) for x in bbox]
    print(bbox)

    (top, left, bottom, right) = bbox
    
    disp = frames[i].copy()
    disp = cv2.rectangle(disp, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.imshow("frame", disp)
    cv2.waitKey()

