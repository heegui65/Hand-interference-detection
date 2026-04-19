# logic.py
import numpy as np
from config import ALARM_FRAME_THRESH

def iou(boxA, boxB):
    """Calculate IOU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)

class InterventionDetector:
    def __init__(self):
        self.frame_count = 0
        self.alarm_flag = False

    def update(self, detections):
        """Increment counter when hand detected, trigger alarm when threshold reached."""
        if len(detections) > 0:
            self.frame_count += 1
        else:
            self.frame_count = 0

        self.alarm_flag = self.frame_count >= ALARM_FRAME_THRESH
        return self.alarm_flag

    def reset(self):
        """Reset alarm state."""
        self.frame_count = 0
        self.alarm_flag = False