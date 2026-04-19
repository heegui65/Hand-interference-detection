import cv2
import numpy as np
from core.infer import HandDetector
from core.postprocess import parse_detections
from core.logic import InterventionDetector
from config import (CROP_X1, CROP_Y1, CROP_X2, CROP_Y2, CROP_SIZE,
                   TARGET_SIZE, ROTATE_ANGLE, VIDEO_SOURCE, ORIGINAL_VIDEO_WIDTH, ORIGINAL_VIDEO_HEIGHT,
                   INPUT_SCALE)

# Display scale (for window display, does not affect detection)
DISPLAY_SCALE = 0.5  # 50% display

def rotate_image(image, angle):
    """Rotate image by specified angle."""
    if angle == 90: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def crop_center_region(frame):
    """Crop center square region from scaled video."""
    crop_x1 = int(CROP_X1 * INPUT_SCALE)
    crop_y1 = int(CROP_Y1 * INPUT_SCALE)
    crop_x2 = int(CROP_X2 * INPUT_SCALE)
    crop_y2 = int(CROP_Y2 * INPUT_SCALE)
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]

def convert_coords_to_display(box_640):
    """Convert coordinates from 640x640 inference space to scaled display space."""
    x1, y1, x2, y2 = box_640
    
    scaled_crop_size = CROP_SIZE * INPUT_SCALE
    crop_scale = scaled_crop_size / TARGET_SIZE
    x1_crop = x1 * crop_scale
    y1_crop = y1 * crop_scale
    x2_crop = x2 * crop_scale
    y2_crop = y2 * crop_scale
    
    crop_x1_scaled = int(CROP_X1 * INPUT_SCALE)
    crop_y1_scaled = int(CROP_Y1 * INPUT_SCALE)
    x1_display = x1_crop + crop_x1_scaled
    y1_display = y1_crop + crop_y1_scaled
    x2_display = x2_crop + crop_x1_scaled
    y2_display = y2_crop + crop_y1_scaled
    
    return [x1_display, y1_display, x2_display, y2_display]

def main():
    detector = HandDetector("best.onnx")
    logic = InterventionDetector()

    if VIDEO_SOURCE is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIGINAL_VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIGINAL_VIDEO_HEIGHT)
        if not cap.isOpened():
            return
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            return

    cv2.namedWindow("Hand Detection (Monitor View)")

    while True:
        ret, frame = cap.read()
        if not ret:
            if VIDEO_SOURCE is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                continue
            else:
                break

        # Resize input resolution immediately to reduce processing
        if INPUT_SCALE != 1.0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * INPUT_SCALE), int(h * INPUT_SCALE)))
        
        frame = rotate_image(frame, ROTATE_ANGLE)
        
        display_frame = frame.copy()
        
        cropped_frame = crop_center_region(frame)
        
        outputs = detector.infer(cropped_frame)
        detections = parse_detections(outputs)
        # print(f"检测到 {len(detections)} 个目标")

        # Temporary debug: check coordinate conversion for first detection
        if len(detections) > 0:
            det = detections[0]
            box_640 = det['box']
            box_display = convert_coords_to_display(box_640)
            print(f"640 space: {box_640}")
            print(f"Display space: {box_display}")
            print(f"Display frame size: {display_frame.shape[1]}x{display_frame.shape[0]}")

        # Draw bounding boxes (coordinates converted back to scaled video space)
        for det in detections:
            box_640 = det['box']
            box_display = convert_coords_to_display(box_640)
            bx1, by1, bx2, by2 = map(int, box_display)
            
            # Check coordinate validity
            if bx1 < bx2 and by1 < by2 and bx1 >= 0 and by1 >= 0 and bx2 <= display_frame.shape[1] and by2 <= display_frame.shape[0]:
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{det['conf']:.2f}", (bx1, max(by1-5, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # Coordinates out of range, but still attempt to draw (boundary case)
                cv2.rectangle(display_frame, (max(0, bx1), max(0, by1)), 
                              (min(display_frame.shape[1], bx2), min(display_frame.shape[0], by2)), 
                              (0, 255, 0), 2)
                cv2.putText(display_frame, f"{det['conf']:.2f}", (max(0, bx1), max(max(by1-5, 20), 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Alarm judgment
        alarm = len(detections) > 0
        if alarm:
            cv2.putText(display_frame, "ALARM! HAND DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Scale display window proportionally
        display_frame = cv2.resize(display_frame, (int(ORIGINAL_VIDEO_WIDTH * INPUT_SCALE * DISPLAY_SCALE), 
                                                     int(ORIGINAL_VIDEO_HEIGHT * INPUT_SCALE * DISPLAY_SCALE)))

        cv2.imshow("Hand Detection (Monitor View)", display_frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()