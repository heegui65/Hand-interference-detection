# postprocess.py
import cv2
import numpy as np
from config import CONF_THRESH, IOU_THRESH, TARGET_SIZE

def parse_detections(outputs):
    """Parse model outputs and return list of detection boxes in pixel coordinates."""
    pred = outputs[0]  # (1,5,8400)
    pred = np.squeeze(pred).transpose(1, 0)  # (8400,5)

    boxes = pred[:, :4]  # cx,cy,w,h in pixel coordinates
    scores = pred[:, 4]  # confidence scores

    # Filter by confidence
    mask = scores > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]

    # Convert to [x1,y1,w,h] for NMS
    nms_boxes = []
    for b in boxes:
        cx, cy, w, h = b
        x1 = cx - w/2
        y1 = cy - h/2
        nms_boxes.append([int(x1), int(y1), int(w), int(h)])

    # Perform NMS
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), CONF_THRESH, IOU_THRESH)

    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            cx, cy, w, h = boxes[idx]
            conf = scores[idx]
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            results.append({
                "box": [x1, y1, x2, y2],
                "conf": float(conf),
                "cls": 0
            })
    return results